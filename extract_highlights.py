import argparse
import copy
import json
import os
import random
from typing import List, Tuple

import nltk
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from context_cite import ContextCiter, Qwen3ContextCiter


input_key = {
    "gov_report": "input",
    "qmsum": "input",
}

output_key = {
    "gov_report": "output",
    "qmsum": "output",
}

MODEL_SHORT_NAMES = {
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-32B": "qwen3-32b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
}

PROMPT_TEMPLATES = {
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report. "
        "You must give your answer in a structured format: \"Summary: [your summary]\", "
        "where [your summary] is your generated summary.\n\nReport:\n{context}"
    ),
    "qmsum": (
        "Read the following meeting transcript. Produce a summary in 4 sentences focusing on key "
        "decisions, action items, and important discussion points. You must give your answer in a "
        "structured format: \"Summary: [your summary]\", where [your summary] is your generated summary.\n"
        "==========\n[MEETING TRANSCRIPT]\n==========\n{context}\nNow generate the summary in 4 sentences:"
    ),
}

MAX_NEW_TOKENS = {"gov_report": 1024, "qmsum": 512}

# --- Shared utilities ---

def load_data(dataset_name):
    dataset_map = {"gov_report": "gov_report", "qmsum": "qmsum"}
    return load_dataset("tau/scrolls", dataset_map[dataset_name])["validation"]


def split_into_sentences(text: str) -> Tuple[List[str], List[str]]:
    sentences = []
    for line in text.splitlines():
        sentences.extend(nltk.sent_tokenize(line))
    separators = []
    cur_start = 0
    for sentence in sentences:
        cur_end = text.find(sentence, cur_start)
        separators.append(text[cur_start:cur_end])
        cur_start = cur_end + len(sentence)
    return sentences, separators


def make_save_path(args) -> str:
    base = f"{args.dataset}_{args.num_samples}samples_{args.num_sents}sents"
    if args.method == "lexrank":
        order = "ordered" if args.keep_order else "ranked"
        filename = f"{base}_lexrank_{order}.json"
    elif args.method == "random":
        filename = f"{base}_random.json"
    elif args.method == "cc":
        short = MODEL_SHORT_NAMES.get(args.model_name, args.model_name.replace("/", "_"))
        filename = f"{base}_cc_{short}.json"
    return os.path.join(args.save_dir, filename)


# --- LexRank ---

def run_lexrank(test_data, args):
    from lexrank import LexRank
    from lexrank.mappings.stopwords import STOPWORDS

    print("Initializing LexRank from training split (this may take a few minutes)...")
    train_data = load_dataset("tau/scrolls", args.dataset, split="train")
    train_documents = [[item["input"]] for item in train_data]
    lxr = LexRank(train_documents, stopwords=STOPWORDS["en"])

    processed_samples = []
    for sample in tqdm(test_data):
        document = sample[input_key[args.dataset]]
        sentences, _ = split_into_sentences(document)
        scores = lxr.rank_sentences(sentences, threshold=None, fast_power_method=False)

        if args.keep_order:
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:args.num_sents]
            selected_indices = sorted(i for i, _ in ranked)
            attributed_sents = [
                {"input_sequence": sentences[i], "score": float(scores[i])} for i in selected_indices
            ]
        else:
            ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:args.num_sents]
            attributed_sents = [{"input_sequence": s, "score": float(sc)} for s, sc in ranked]

        processed_sample = copy.deepcopy(sample)
        processed_sample["attributed_sents"] = attributed_sents
        processed_samples.append(processed_sample)

    return processed_samples


# --- Random ---

def run_random(test_data, args):
    processed_samples = []
    for sample in tqdm(test_data):
        document = sample[input_key[args.dataset]]
        sentences, _ = split_into_sentences(document)

        n = args.num_sents
        if n >= len(sentences):
            selected = sentences
        else:
            selected = [sentences[i] for i in sorted(random.sample(range(len(sentences)), n))]

        attributed_sents = [{"input_sequence": s, "score": 1.0} for s in selected]
        processed_sample = copy.deepcopy(sample)
        processed_sample["attributed_sents"] = attributed_sents
        processed_samples.append(processed_sample)

    return processed_samples


# --- ContextCite ---

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    context_window_length = getattr(
        config, "max_position_embeddings", getattr(config, "n_positions", None)
    )
    kwargs = dict(device_map="auto")
    kwargs["torch_dtype"] = "auto" if "Qwen" in model_name else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = context_window_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_cc(test_data, args):
    model, tokenizer = load_model(args.model_name)

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS[args.dataset],
        "do_sample": False,
        "temperature": 0.0,
        "use_cache": True,
    }
    if "Llama-3" in args.model_name:
        generate_kwargs["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    processed_samples = []
    skipped_samples = []

    for idx, sample in tqdm(enumerate(test_data)):
        context = sample[input_key[args.dataset]]
        try:
            CiterClass = Qwen3ContextCiter if "Qwen3" in args.model_name else ContextCiter
            cc = CiterClass(model, tokenizer, context, query="", num_ablations=args.num_ablations)
            cc.prompt_template = PROMPT_TEMPLATES[args.dataset]
            cc.generate_kwargs = generate_kwargs
            results = cc.get_attributions(as_dataframe=True, top_k=args.num_sents, verbose=False)
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM at sample {idx}, skipping...")
            torch.cuda.empty_cache()
            skipped_samples.append(dict(sample))
            continue

        attributed_sents = [
            {"input_sequence": row["Source"], "score": row["Score"]}
            for _, row in results.data.iterrows()
        ]
        processed_sample = copy.deepcopy(sample)
        processed_sample["attributed_sents"] = attributed_sents
        processed_sample["generated_summary"] = cc.response
        processed_samples.append(processed_sample)

    return processed_samples, skipped_samples


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lexrank", "random", "cc"])
    parser.add_argument("--dataset", required=True, choices=["gov_report", "qmsum"])
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--num_sents", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="results/highlights")
    # LexRank only
    parser.add_argument("--keep_order", action="store_true",
                        help="Keep extracted sentences in original document order")
    # ContextCite only
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        choices=list(MODEL_SHORT_NAMES.keys()))
    parser.add_argument("--num_ablations", type=int, default=64)
    parser.add_argument("--shard", type=int, default=None,
                        help="Process shard N of the data (each shard has --num_samples instances)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(".env")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)

    test_data = load_data(args.dataset)

    if args.method == "cc" and args.shard is not None:
        start = args.shard * args.num_samples
        end = min(start + args.num_samples, len(test_data))
        test_data = test_data.select(range(start, end))
    else:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))

    if args.method == "lexrank":
        processed_samples = run_lexrank(test_data, args)
    elif args.method == "random":
        processed_samples = run_random(test_data, args)
    elif args.method == "cc":
        processed_samples, skipped_samples = run_cc(test_data, args)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = make_save_path(args)
    with open(save_path, "w") as f:
        json.dump(processed_samples, f, indent=4)
    print(f"Saved {len(processed_samples)} samples → {save_path}")

    if args.method == "cc" and skipped_samples:
        skipped_path = save_path.replace(".json", "_skipped.json")
        with open(skipped_path, "w") as f:
            json.dump(skipped_samples, f, indent=4)
        print(f"Skipped {len(skipped_samples)} samples → {skipped_path}")


if __name__ == "__main__":
    main()
