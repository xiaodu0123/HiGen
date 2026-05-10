import argparse
import json
import os
import re

from datasets import load_dataset
from dotenv import load_dotenv
from vllm import LLM, SamplingParams


MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-32B": "qwen3-32b",
}

DATASET_MAP = {"gov_report": "gov_report", "qmsum": "qmsum"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate summaries on long context datasets")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=list(MODEL_SHORT_NAMES.keys()))
    parser.add_argument("--dataset", required=True, choices=["gov_report", "qmsum"])
    parser.add_argument("--method", default="base", choices=["base", "higen", "sum_cot"])
    parser.add_argument("--save-path", default="results/summary")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=80000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--highlight-data", type=str, default=None,
                        help="Path to the extracted sentence highlights")
    parser.add_argument("--highlight-type", type=str, default="sent",
                        help="Highlight method used in the output filename")
    return parser.parse_args()


def get_prompt(doc, important_sents, method, dataset):
    if method == "base":
        prompt_template = {
            "qmsum": (
                "Read the following meeting transcript. Produce a summary in 4 sentences focusing on key "
                "decisions, action items, and important discussion points. You must give your answer in a "
                "structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n"
                "==========\n[MEETING TRANSCRIPT]\n==========\n"
                f"{doc}\nNow generate the summary in 4 sentences: "
            ),
            "gov_report": (
                "You are given a report by a government agency. Write a one-page summary of the report. "
                "You must give your answer in a structured format: \"Summary: [your summary]\", "
                f"where [your summary] is your generated summary.\n\nReport:\n{doc}"
            ),
        }

    elif method == "higen":
        key_points = "\nYou should only focus on the following key points:\n" + \
                     "\n".join(f"{i+1}. {s}" for i, s in enumerate(important_sents)) + "\n"
        prompt_template = {
            "qmsum": (
                "Read the following meeting transcript. Produce a summary in 4 sentences focusing on key "
                "decisions, action items, and important discussion points. You must give your answer in a "
                "structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n"
                "==========\n[MEETING TRANSCRIPT]\n==========\n"
                f"{doc}\n{key_points}\nNow generate the summary in 4 sentences: "
            ),
            "gov_report": (
                "You are given a report by a government agency. Write a one-page summary of the report "
                "focusing on the main points. You must give your answer in a structured format: \"Summary: "
                "[your summary]\", where [your summary] is your generated summary.\n\n"
                f"Report:\n{doc}\n{key_points}"
            ),
        }

    elif method == "sum_cot":
        questions = (
            "1. What are the important entities in this document?\n"
            "2. What are the important dates in this document?\n"
            "3. What events are happening in this document?\n"
            "4. What is the result of these events?\n"
        )
        prompt_template = {
            "qmsum": (
                "Read the following meeting transcript. Answer the questions below and then produce a summary "
                "in 4 sentences by integrating the information in your answers. You must give your response in "
                "a structured format: \"Answers:\n1. answer1, 2. answer2, ...\nSummary: [your summary]\", "
                "where [your summary] is your generated summary.\n"
                "==========\n[MEETING TRANSCRIPT]\n==========\n"
                f"{doc}\n{questions}"
            ),
            "gov_report": (
                "You are given a report by a government agency. Answer the questions below and then write a "
                "one-page summary of the report by integrating the information in your answers. You must give "
                "your response in a structured format: \"Answers:\n1. answer1, 2. answer2, ...\nSummary: "
                "[your summary]\", where [your summary] is your generated summary.\n\n"
                f"Report:\n{doc}\n\nQuestions:\n{questions}"
                "Provide short answers containing only the most relevant entities to the questions without "
                "using bullet points and then generate the summary after \"Summary\" prompt word: "
            ),
        }

    return prompt_template[dataset]


def extract_summary(text):
    if "Summary:" in text:
        summary = text[text.find("Summary:") + len("Summary:"):].strip()
        return summary.strip("**\n\n")
    return text.strip()


def extract_answers_and_summary(text):
    summary_start = text.find("Summary:")
    if summary_start == -1:
        return [], ""

    answers_text = text[:summary_start].strip("\n[]")
    summary = text[summary_start + len("Summary:"):].strip()

    answers = []
    for line in answers_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.", line):
            answer = re.sub(r"^\d+\.\s*", "", line).strip()
        elif line.startswith("- ") or line.startswith("* "):
            answer = line[2:].strip()
        else:
            continue
        if answer:
            answers.append(answer)

    return answers, summary


def main():
    args = parse_args()
    load_dotenv(".env")
    if args.method == "higen" and not args.highlight_data:
        raise ValueError("--method higen requires --highlight-data")

    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=False,
    )
    if "Qwen" in args.model:
        llm_kwargs["hf_overrides"] = {
            "rope_parameters":{
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
        }
        llm_kwargs["trust_remote_code"] = True
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    if args.highlight_data:
        with open(args.highlight_data) as f:
            data = json.load(f)[:args.max_samples]
    else:
        dataset = load_dataset("tau/scrolls", DATASET_MAP[args.dataset])[args.split]
        data = dataset.select(range(min(args.max_samples, len(dataset))))

    chat_messages = []
    for item in data:
        attributed_sents = (
            [s["input_sequence"] for s in item["attributed_sents"] if s["score"] > 0]
            if args.highlight_data else None
        )
        prompt = get_prompt(item["input"], attributed_sents, args.method, args.dataset)
        chat_messages.append([
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": prompt},
        ])

    chat_kwargs = dict(messages=chat_messages, sampling_params=sampling_params, use_tqdm=True)
    if "Qwen3" in args.model:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = llm.chat(**chat_kwargs)

    results = []
    for item, output in zip(data, outputs):
        raw = output.outputs[0].text
        result_item = dict(item)
        if args.method == "sum_cot":
            answers, summary = extract_answers_and_summary(raw)
            result_item["answers"] = answers
            result_item["generated_summary"] = summary
            result_item["raw_output"] = raw
        else:
            result_item["generated_summary"] = extract_summary(raw)
        results.append(result_item)

    short_name = MODEL_SHORT_NAMES.get(args.model, args.model.replace("/", "_"))
    if args.highlight_data:
        filename = f"{short_name}_{args.dataset}_{args.split}_{len(results)}_highlights-{args.highlight_type}_{args.method}.json"
    else:
        filename = f"{short_name}_{args.dataset}_{args.split}_{len(results)}_{args.method}.json"

    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, filename)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} summaries → {save_path}")


if __name__ == "__main__":
    main()
