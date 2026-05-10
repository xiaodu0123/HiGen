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

PROMPT_TEMPLATES = {
    "qmsum": (
        "Read the following meeting transcript. Extract a list of {num_sents} key sentences from the "
        "input document and then produce a summary in 4 sentences only focusing on the extracted sentences. "
        "You must give your answer in a structured format: \"Key Sentences:\n1. sentence1, 2. sentence2, "
        "...\nSummary: [your summary]\", where [your summary] is your generated summary.\n"
        "==========\n[MEETING TRANSCRIPT]\n==========\n{doc}"
    ),
    "gov_report": (
        "You are given a report by a government agency. Extract a list of {num_sents} key sentences from "
        "the input document and then write a one-page summary of the report only focusing on the extracted "
        "sentences. You must give your answer in a structured format: \"Key Sentences:\n1. sentence1, "
        "2. sentence2, ...\nSummary: [your summary]\", where [your summary] is your generated summary."
        "\n\nReport:\n{doc}"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract sentence highlights and generate summaries")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=list(MODEL_SHORT_NAMES.keys()))
    parser.add_argument("--dataset", required=True, choices=["gov_report", "qmsum"])
    parser.add_argument("--save-path", default="results/highlights")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=80000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--num-sents", type=int, default=30, help="Number of key sentences to extract")
    
    return parser.parse_args()


def extract_sentences_and_summary(text):
    summary_start = text.find("Summary:")
    if summary_start == -1:
        return [], ""

    key_sentences_text = text[:summary_start].strip("\n[]")
    summary = text[summary_start + len("Summary:"):].strip()

    sentences = []
    for line in key_sentences_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.", line):
            sentence = re.sub(r"^\d+\.\s*", "", line).strip()
        elif line.startswith("- ") or line.startswith("* "):
            sentence = line[2:].strip()
        else:
            continue
        if sentence:
            sentences.append(sentence)

    return sentences, summary


def main():
    args = parse_args()
    load_dotenv(".env")

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

    # dataset = load_dataset("tau/scrolls", args.dataset, trust_remote_code=True)[args.split]
    dataset = load_dataset("tau/scrolls", args.dataset)[args.split]
    data = dataset.select(range(min(args.max_samples, len(dataset))))

    chat_messages = [
        [
            {"role": "system", "content": "You are an expert summarization assistant."},
            {"role": "user", "content": PROMPT_TEMPLATES[args.dataset]
                .replace("{num_sents}", str(args.num_sents))
                .replace("{doc}", item["input"])},
        ]
        for item in data
    ]

    chat_kwargs = dict(messages=chat_messages, sampling_params=sampling_params, use_tqdm=True)
    if "Qwen3" in args.model:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    outputs = llm.chat(**chat_kwargs)

    results = []
    for item, output in zip(data, outputs):
        prediction = output.outputs[0].text
        key_sentences, summary = extract_sentences_and_summary(prediction)
        attributed_sents = [
            {"input_sequence": sent.strip(), "score": 1.0}
            for sent in key_sentences if sent.strip()
        ]
        result_item = dict(item)
        result_item["attributed_sents"] = attributed_sents
        result_item["generated_summary"] = summary
        result_item["raw_output"] = prediction
        results.append(result_item)

    short_name = MODEL_SHORT_NAMES.get(args.model, args.model.replace("/", "_"))
    filename = f"{short_name}_{args.dataset}_{args.split}_{len(results)}_highlights_num{args.num_sents}.json"
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, filename)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} samples → {save_path}")


if __name__ == "__main__":
    main()
