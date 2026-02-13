# SeekBench Toolkit

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2509.22391-b31b1b.svg)](https://arxiv.org/abs/2509.22391)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)


## Highlights

- End-to-end evaluation pipeline from raw QA data to analysis figures.
- Annotation ontology for `groundness`, recovery behavior, and calibration.
- Three paper-aligned analysis scripts:
  - `analysis/grounded_reason.py`
  - `analysis/recovery.py`
  - `analysis/calibration.py`

## Project Structure

```text
search_evaluation_toolkit/
├── trace_eval/
│   ├── generate_traces.py
│   └── rerun_traces.py
├── annotation/
│   ├── parser.py
│   ├── main.py
│   ├── prompts.py
│   └── ontology.json
├── analysis/
│   ├── grounded_reason.py
│   ├── recovery.py
│   ├── calibration.py
│   └── README.md
├── scripts/
│   ├── run_generate_traces.sh
│   └── run_rerun_traces.sh
├── data/
│   └── seekbench_data.jsonl
├── requirements.txt
└── setup.py
```

## Installation

```bash
cd search_evaluation_toolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For annotation APIs:

```bash
export OPENAI_API_KEY=YOUR_KEY
# optional
export OPENAI_BASE=YOUR_OPENAI_COMPATIBLE_ENDPOINT
```

## Quickstart (Included Data)

Run all three analyses on the bundled annotated dataset:

```bash
python analysis/grounded_reason.py --input data/seekbench_data.jsonl --outdir outputs/groundness
python analysis/recovery.py data/seekbench_data.jsonl outputs/recovery
python analysis/calibration.py --input data/seekbench_data.jsonl --outdir outputs/calibration
```

## Workflows

### A) You need to generate traces

1. Generate raw traces from a QA parquet dataset.
2. Parse raw trace text to structured `trace` steps.
3. Annotate each step.
4. Run analysis scripts.

```bash
# 1) Generate traces
python trace_eval/generate_traces.py \
  --model_path Qwen/Qwen2.5-7B-Instruct \
  --data_path data/hotpotqa/process_test.parquet \
  --output_path outputs/raw_traces.jsonl \
  --do_search

# 2) Parse raw traces
python annotation/parser.py outputs/raw_traces.jsonl -o outputs/parsed_traces.jsonl

# 3) Annotate traces
python annotation/main.py \
  --input_file outputs/parsed_traces.jsonl \
  --output_file outputs/annotated_traces.jsonl \
  --ontology_file annotation/ontology.json \
  --model gpt-4.1-mini \
  --concurrency 30

# 4) Analyze
python analysis/grounded_reason.py --input outputs/annotated_traces.jsonl --outdir outputs/groundness
python analysis/recovery.py outputs/annotated_traces.jsonl outputs/recovery
python analysis/calibration.py --input outputs/annotated_traces.jsonl --outdir outputs/calibration
```

### B) You already have traces

- If traces contain only raw `sequences_str`, run `annotation/parser.py` first.
- If traces are structured but not annotated, run `annotation/main.py`.
- If traces are already annotated, run analysis scripts directly.

## Input/Output Reference

### `trace_eval/generate_traces.py`

- Input:
  - `--data_path`: QA parquet file.
  - model arguments (`--model_path` or checkpoint args).
- Output:
  - `--output_path`: JSONL traces with fields like `question`, `ground_truth`, `sequences_str`, `reward`, `do_search`.

### `trace_eval/rerun_traces.py`

- Input:
  - positional `input_file`: JSONL traces.
- Output:
  - positional `output_file`: JSONL with regenerated answer/reward fields.

### `annotation/parser.py`

- Input:
  - positional `input_file`: raw trace JSONL.
- Output:
  - `-o/--output_file`: structured JSONL with parsed `trace` steps.

### `annotation/main.py`

- Input:
  - `--input_file`: structured traces JSONL.
  - `--ontology_file`: annotation ontology JSON.
- Output:
  - `--output_file`: annotated JSONL with per-step `annotation` fields (including `groundness`).

### Analysis scripts (`analysis/grounded_reason.py`, `analysis/recovery.py`, `analysis/calibration.py`)

- Input:
  - annotated trace JSONL (typically output of `annotation/main.py`).
- Output:
  - figures (PNG) and summary tables (CSV where applicable) under the specified output directory.

## `data/` Folder

`data/seekbench_data.jsonl` is the included annotated dataset used for quick analysis and reproducibility checks.

Typical fields include:
- sample metadata: `question`, `ground_truth`, `dataset`, `model`, `id`
- trajectory: `trace` (step list)
- outcome: `reward`, `is_correct`, `final_answer`, `queries`

## `scripts/` Folder

- `scripts/run_generate_traces.sh`: batch launcher for trace generation.
- `scripts/run_rerun_traces.sh`: batch launcher for rerunning existing traces.

Both scripts are path-safe and can be run from repository root or toolkit root.

## Reproducibility Notes

- Use the same model, prompt template, and retrieval settings when comparing runs.
- Keep annotation model and ontology fixed for fair comparisons.
- Analysis scripts expect annotation keys to use `groundness` terminology.

## Citation

```
@inproceedings{
shao2026do,
title={Do {LLM} Agents Know How to Ground,  Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents},
author={Jiaqi Shao and Yuxiang Lin and Munish Prasad Lohani and Yufeng Miao and Bing Luo},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=r0L9GwlnzP}
}
```