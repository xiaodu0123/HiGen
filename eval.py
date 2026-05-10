import argparse
import copy
import json
import os
from collections import defaultdict

import nltk
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import evaluate as hf_evaluate
from summac.model_summac import SummaCConv
from fact_score.atomic_facts import AtomicFactGenerator
from fact_score.factscorer import FactScorer


input_key = {"gov_report": "input", "qmsum": "input"}
output_key = {"gov_report": "output", "qmsum": "output"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated summaries")
    parser.add_argument("--data_path", required=True, type=str,
                        help="Path to JSON file with generated summaries")
    parser.add_argument("--dataset", required=True, choices=["gov_report", "qmsum"])
    parser.add_argument("--metrics", nargs="+", required=True,
                        choices=["rouge", "bert", "summac", "factscore"],
                        help="Metric values to compute")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Output directory for saving evaluation metrics and caches")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="OpenAI model used to compute FactScore")
    return parser.parse_args()


def postprocess_text(preds, labels):
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    return preds, labels


def compute_rouge(pred, gold):
    rouge = hf_evaluate.load("rouge")
    proc_pred, proc_gold = postprocess_text([pred], [gold])
    scores = rouge.compute(predictions=proc_pred, references=proc_gold)
    return {
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "rougeLsum": scores["rougeLsum"],
    }


def compute_bert(pred, gold):
    bertscore = hf_evaluate.load("bertscore")
    result = bertscore.compute(
        predictions=[pred], references=[gold],
        model_type="microsoft/deberta-xlarge-mnli", lang="en",
    )
    return {
        "bertscore_p": result["precision"][0],
        "bertscore_r": result["recall"][0],
        "bertscore_f1": result["f1"][0],
    }


def compute_summac(doc, pred, model_conv):
    result = model_conv.score([doc], [pred])
    return {"summac_score": result["scores"][0]}


def filter_facts(facts):
    bad_substrings = [
        'someone', 'something', 'somebody',
        'is a person', 'is a character', 'are people', 'are characters',
    ]
    facts = [f for f in facts if f == '<MALFORMED SENTENCE>' or len(f.split()) > 2]
    facts = [f for f in facts if f == '<MALFORMED SENTENCE>' or
             not any(x in f.lower() for x in bad_substrings)]
    facts = [f for f in facts if 'is mentioned' not in f.lower() and
             'are mentioned' not in f.lower() and
             'is there' not in f.lower() and
             'are there' not in f.lower()]
    facts = [f for f in facts if f == '<MALFORMED SENTENCE>' or not f.endswith(' to')]
    return facts


def compute_factscore(pred, doc, afg, fs, summname):
    facts_and_sources = afg.extract_facts(pred)
    predicted_facts = filter_facts([x for item in facts_and_sources for x in item[1]])
    fact_precision, score_per_fact = fs.get_score(predicted_facts, doc, summname=summname)
    if len(predicted_facts) == 0 or np.isnan(fact_precision):
        fact_precision = 0.0
    return {"fact_precision": float(fact_precision)}, predicted_facts, score_per_fact


def main():
    args = parse_args()
    load_dotenv(".env")

    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    exp_dir = args.exp_dir or os.path.join("results/metrics", stem)
    os.makedirs(exp_dir, exist_ok=True)

    with open(args.data_path) as f:
        data = json.load(f)

    model_conv = None
    if "summac" in args.metrics:
        model_conv = SummaCConv(
            models=["vitc"], bins="percentile", granularity="sentence",
            nli_labels="e", device="cuda:0", start_file="default", agg="mean",
        )

    afg = fs = None
    if "factscore" in args.metrics:
        model_short = args.model_name.replace("/", "_")
        factscore_cache_dir = os.path.join(exp_dir, model_short)
        afg = AtomicFactGenerator(model_name=args.model_name, cache_dir_prefix=factscore_cache_dir)
        fs = FactScorer(model_name=args.model_name, cache_dir_prefix=factscore_cache_dir)

    scores = defaultdict(list)
    annotated_samples = []

    for idx, sample in tqdm(enumerate(data)):
        doc = sample[input_key[args.dataset]]
        gold = sample[output_key[args.dataset]]
        pred = sample["generated_summary"]
        result_item = copy.deepcopy(sample)

        if "rouge" in args.metrics:
            m = compute_rouge(pred, gold)
            result_item.update(m)
            for k, v in m.items():
                scores[k].append(v)

        if "bert" in args.metrics:
            m = compute_bert(pred, gold)
            result_item.update(m)
            for k, v in m.items():
                scores[k].append(v)

        if "summac" in args.metrics:
            m = compute_summac(doc, pred, model_conv)
            result_item.update(m)
            for k, v in m.items():
                scores[k].append(v)

        if "factscore" in args.metrics:
            m, predicted_facts, score_per_fact = compute_factscore(
                pred, doc, afg, fs, summname=f"sample_{idx}"
            )
            result_item.update(m)
            result_item["predicted_facts"] = predicted_facts
            result_item["score_per_fact"] = score_per_fact
            for k, v in m.items():
                scores[k].append(v)

        annotated_samples.append(result_item)

    avg_metrics = {"exp_name": stem}
    for k, v in scores.items():
        avg_metrics[k] = sum(v) / len(v)

    print(json.dumps(avg_metrics, indent=2))

    with open(os.path.join(exp_dir, "average_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    with open(os.path.join(exp_dir, "annotated_samples.json"), "w") as f:
        json.dump(annotated_samples, f, indent=4)

    print(f"Results saved to {exp_dir}/")


if __name__ == "__main__":
    main()
