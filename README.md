# Enhancing Long Document Long Form Summarisation with Self-Planning
This repository contains the code implementation of Highlight-guided Generation approach (HiGen), proposed in the AACL'25 paper: [Enhancing Long Document Long Form Summarisation with Self-Planning](https://arxiv.org/pdf/2512.17179). HiGen is a self-planning summarisation framework for long document summarisation that leverages sentence-level highlights derived from the input document to guide the summary generation.  

## Environment Setup
1. Install torch and vLLM (tested with vLLM 0.19.0)
2. Install dependences required for evaluation
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt punkt_tab names
   ```
Note: `summac` library requires an older version of `transformers`, which can be incompatible with `vLLM`. A simple workaround is to create separate virtual environments for inference and evaluation.

3. Set up OpenAI API key and Hugging Face token in `.env` file in the project root path.

    Template of `.env` file
    ```
    OPENAI_API_KEY="add your API key"
    HF_TOKEN="add your token"
    ```
    Instead, you can set up environment variables
    ```
    export OPENAI_API_KEY="your API key"
    export HF_TOKEN="your huggingface token"
    ```

## Project Structure
The code base is organised in the following structure:
- `context_cite/`: contains implementation of ContextCite attribution (adapted from [ContextCite repo](https://github.com/MadryLab/context-cite))
- `fact_score/`: contains implementation of FactScore metric (adapted from [PRISMA metric](https://github.com/lou1sm/modular_multimodal_summarization))
- `extract_highlights.py`: implement baseline approaches for extracting sentence highlights (including ContextCite attribution, LexRank score and random sentences)
- `gen_highlights.py`: use LLM to extract sentence highlights and generate summaries
- `gen_summary.py`: generate summaries guided by the extracted highlights
- `eval.py`: compute evaluation metrics for the generated summaries (including ROUGE-L, BertScore, Summa-C and FactScore)

## Experiments
We evaluate the performance of HiGen on GovReport and QMSum datasets. We use the version of the datasets provided in [SCROLLS benchmark](https://www.scrolls-benchmark.com/), available on [Hugging Face](https://huggingface.co/datasets/tau/scrolls). 

### Generate summaries using HiGen
Extract sentence-level highlights from the input documents:
```bash
python gen_highlights.py \
    --model Qwen/Qwen3-8B \
    --dataset gov_report \
    --max-samples 300 \
    --num-sents 30 \
    --max-tokens 4000 \
    --max-model-len 80000 \
    --save-path results/highlights
```
Generate highlight-guided summaries:
```bash
python gen_summary.py \
    --model Qwen/Qwen3-8B \
    --dataset gov_report \
    --method higen \
    --max-samples 300 \
    --max-tokens 1024 \
    --max-model-len 80000 \
    --highlight-data ${PATH to extracted highlights} \
    --highlight-type sent \
    --save-path results/summary
```
- Set the path to the JSON file with extracted highlight sentences by `--highlight-data`

### Run baseline experiments
Extract highlights using [ContextCite attribution](https://proceedings.neurips.cc/paper_files/paper/2024/file/adbea136219b64db96a9941e4249a857-Paper-Conference.pdf):
```bash
python extract_highlights.py \
    --method cc \
    --dataset gov_report \
    --num_samples 300 \
    --num_sents 30 \
    --model_name Qwen/Qwen3-8B \
    --save_dir results/highlights
```

Extract highlights using [LexRank score](https://github.com/crabcamp/lexrank):
```bash
python extract_highlights.py \
    --method lexrank \
    --dataset gov_report \
    --num_samples 300 \
    --num_sents 30 \
    --keep_order \
    --save_dir results/highlights
```
- The number of sentences to extract can be set by `--num_sents`. We used 30 highlighted sentences in our experiments in the paper. 

### Compute evaluation metrics
```bash
python eval.py \
    --data_path ${PATH to prediction file} \
    --dataset gov_report \
    --metrics rouge bert summac factscore \
    --model_name gpt-4o-mini \
    --exp_dir results/metrics
```
- `--model_name`: Specify which model is used for computing FactScore
- `--metrics`: Specify which metrics to compute

## Citation
```bibtex
@inproceedings{du-etal-2025-enhancing,
    title = "Enhancing Long Document Long Form Summarisation with Self-Planning",
    author = "Du, Xiaotang  and
      Saxena, Rohit  and
      Perez-Beltrachini, Laura  and
      Minervini, Pasquale  and
      Titov, Ivan",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-short.27/",
    doi = "10.18653/v1/2025.ijcnlp-short.27",
    pages = "317--332",
    ISBN = "979-8-89176-299-2"
}
```