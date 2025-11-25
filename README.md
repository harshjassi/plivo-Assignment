# PII NER — README

## Overview

This repository implements a token-level Named Entity Recognition (NER) system for detecting PII in noisy Speech-To-Text (STT) transcripts.
The system is precision-focused for PII categories and returns character-span entities with PII flags.

**Detected entity types**

* `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`, `CITY`, `LOCATION`

**PII flag**

* `pii = true` for: `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`
* `pii = false` for: `CITY`, `LOCATION`

This fork rewrites all training / inference / evaluation scripts into an optimization-lean, production-friendly layout and adds conservative post-processing to improve PII precision.

---

## Repository layout

```
PII_NER_ASSIGNMENT_IITB/
├─ data/
│  ├─ train.jsonl
│  ├─ dev.jsonl
│  ├─ stress.jsonl
│  └─ test.jsonl
├─ src/
│  ├─ dataset.py          # tokenizer alignment, dataset + collate
│  ├─ model.py            # model builder wrapper
│  ├─ train.py            # training script
│  ├─ predict.py          # inference + BIO->span decoding + validators
│  ├─ eval_span_f1.py     # span-level evaluation & PII aggregation
│  ├─ measure_latency.py  # p50/p95 measurement (batch_size=1)
│  └─ labels.py           # label lists & PII mapping
├─ requirements.txt
└─ README.md
```

---

## Requirements

Recommended Python version: **3.8+** (tested on 3.10/3.11).
Install dependencies:

```bash
pip install -r requirements.txt
```

Typical contents of `requirements.txt`:

* `transformers`
* `torch`
* `tqdm`

(Use a virtualenv or conda env.)

---

## Quick usage (single-line CLI commands)

> **Note for Windows PowerShell:** do **not** use `\` for multiline. Put arguments on a single line or use backtick `` ` `` for multiline.

### 1) Train

```bash
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out
```

### 2) Predict (inference)

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
```

### 3) Evaluate

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

### 4) Latency measure

```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50 --device cpu
```

---

## Input / output formats

### Input (train/dev JSONL)

Each line: one JSON object

```json
{
  "id": "utt_0012",
  "text": "my credit card number is four four four four ...",
  "entities": [
    {"start": 3, "end": 19, "label": "CREDIT_CARD"},
    ...
  ]
}
```

`text` is noisy STT style (spelled digits, “dot”, “at”, filler words).

### Predict output

`predict.py` writes a JSON mapping `id -> list of entities`:

```json
{
  "utt_0012": [
    {"start": 3, "end": 19, "label": "CREDIT_CARD", "pii": true},
    ...
  ],
  ...
}
```

Start/end are **character offsets** on the original transcript.

---

## Implementation notes & design choices

### Model

* Primary detector: learned token classifier (distilBERT by default). This meets the requirement to use a learned sequence labeler.
* `src/model.py` builds a token-classification model and supplies label maps.

### Dataset & token-to-char alignment

* `src/dataset.py` creates character-level BIO tags and maps them to tokenizer offsets using offset mapping. A safe fallback is used when lengths mismatch.

### Decoding & post-processing

* `src/predict.py` decodes BIO token labels into character spans, then runs **conservative validators**:

  * `CREDIT_CARD`: check digit-only length in [13, 19]
  * `PHONE`: check digit-only length in [7, 15]
  * `EMAIL`: accept `@` & `.` patterns or spoken `at`/`dot` patterns and normalize `at` → `@` and `dot` → `.`
* These conservative filters are intentionally strict to boost **PII precision** (even at the cost of recall), matching the assignment focus.

### Evaluation

* `src/eval_span_f1.py` computes:

  * Per-entity P/R/F1 (exact span matches)
  * Macro F1 across entity types
  * Aggregated P/R/F1 for PII vs non-PII spans

### Latency measurement

* `src/measure_latency.py` runs forward passes on batch size 1 and reports p50 & p95 in milliseconds. Warmup steps are included.

---

## Tips to improve latency (p95 ≤ 20 ms CPU)

* If default `distilbert` p95 is too large:

  * Use **dynamic quantization** with PyTorch (`torch.quantization.quantize_dynamic`) for CPU.
  * Export to **ONNX** and run with **ONNXRuntime**, which often yields lower CPU latency.
  * Replace with a smaller tagger (TinyBERT or BiLSTM tagger with frozen embeddings) if strong constraints exist.
* The codebase is organized so adding a TorchScript / ONNX export script is straightforward.

---

## Reproducibility & debugging

* If you see `FileNotFoundError`, check you are running commands from repository root and reference `src/<script>.py` (scripts live under `src/`).
* Windows PowerShell users: run the full command on one line or use backtick `` ` `` for multiline; do not use `\`.
* If training fails due to CUDA issues, try `--device cpu` or ensure CUDA is available.

---

## Common troubleshooting

* **SyntaxError from edited files**: check you saved the file exactly as provided and there are no leftover stray characters.
* **Tokenizer/model loading**: if loading pre-trained models fails due to network restrictions, you can pre-download models with the `transformers` cache or use a local model path for `--model_name`.
* **Length/offset mismatches**: dataset code contains fallbacks; inspect `data/*.jsonl` examples to ensure `start/end` offsets are correct.

---

## What I changed (high level)

* Rewrote all core scripts with:

  * Different function/class names and file-local mappings
  * Cleaner modular structure
  * Conservative post-processing for PII classes
  * Readiness for CPU optimizations
* CLI behavior and file formats preserved so existing evaluation commands still work.

---

## Next steps / optional additions

If you want I can:

* Add a `scripts/run_all.ps1` or `run_all.bat` to run train → predict → eval → latency in sequence.
* Add a small ONNX export + inference wrapper to reduce p95.
* Add a data augmentation script to synthesize noisy STT dev examples (helpful if you need to expand dev set).
* Tune validators to increase recall while maintaining precision.
