# Option-order exchangeability reproduction

This repository contains the compact analysis reproduction for manuscript v126. It reproduces the retained appendix tables and numerical inputs to the paper's figures from compact data.

Run:

```sh
python3 reproduce.py
```

The script uses only the Python standard library and writes `results.json`. It does not download benchmarks, rescore language models, regenerate intervention text or redistribute model weights.

## Files

- `reproduce.py`: all retained calculations and verification checks.
- `data/pooled_items.csv`: reference-model outcomes and screen flags for 14,191 items.
- `data/model_items.csv`: outcomes for seven models on 2,043 shared items.
- `data/intervention_items.csv`: outcomes for six matched arms on 1,000 source items.
- `data/census_benchmarks.csv`: six-benchmark option-order census.
- `data/census_mmlu.csv`: 57-subject MMLU census.
- `results.json`: generated paper-facing output; safe to delete and recreate.

## Scope

The CSV files contain compact sufficient statistics derived from preserved research artifacts. The model-based files retain one row per analyzed item or model-item pair instead of all 24 permutation scores; the census files contain the aggregate rows reported in the paper. The larger raw archive remains separate evidence and is not replaced by this repository.

The public September repository predates the paper's current 1,000-item intervention. This reproduction uses the preserved 10 November intervention result reported by v126. The submitted paper's older 71-item intervention and separate 1,577-item stem experiment are excluded because neither appears in the current paper.

The dataset sources are MedQA, MMLU, MedMCQA, ARC-Challenge, OpenBookQA and CommonsenseQA. The scored models are GPT-2, GPT-2-large, GPT-2-XL, Qwen2.5-1.5B, Qwen2.5-3B, BioGPT-Large and BioMedLM. See the manuscript bibliography for source citations.

A language-model assistant helped write the reproduction code and documentation.
