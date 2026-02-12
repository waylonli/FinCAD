# look-ahead-bias

This repository mitigates **look-ahead bias** in LLM-based financial backtesting—where models use pretrained knowledge of future outcomes (e.g., known winners or crises) instead of relying strictly on information available at the historical time. The codebase provides **two inference-time control methods**, benchmarking pipelines, and cluster scripts for reproducible experiments. This README is written as a **handover document** for a new maintainer.

## Project objective

Build practical, inference-time mechanisms that:
- **Suppress pretrained “future knowledge”** leakage in finance tasks.
- Preserve general benchmark performance as a sanity check.
- Provide **reproducible experiments** with logs and per-item outputs.

## What’s in the repo

**1) Activation steering (legacy / optional)**
- Module: `steering/`
- Goal: steer the model away from “memory recall” and toward “reasoning” using probe-derived vectors.
- Profiles: `recall_suppression` and `entity_defocus`.
- Still available, but CAD is the current research focus.

**2) Context-Aware Decoding (CAD)**
- Module: `cad/`
- Goal: force generation to depend on provided context by subtracting a biased “prior” from the context-conditioned logits.
- Supports **Bias-Amplified CAD** and **Entity-Adaptive α** via `CADCalibrator`.

**3) Benchmarks**
- Folder: `benchmark/`
- Benchmarks implemented: GSM8K, MMLU-Pro, MATH-500, HumanEval.
- Each script supports baseline, steering, and CAD via `--decoding-mode`.

**4) Cluster scripts**
- Folder: `scripts/`
- Run scripts under:
  - `scripts/Eddie/general_benchmark/qwen2_5/`
  - `scripts/lightspeed/general_benchmark/qwen2_5/`

## Key CAD ideas in this repo

**Standard CAD**  
Combined logits:  
`((1 + α) * logits(context)) - (α * logits(prior))`

**Bias-Amplified CAD (novel contribution)**  
The “prior” is not neutral. We explicitly **trigger cheating** in the prior prompt and subtract it, targeting memorized winners or historical outcomes more directly.

**Entity-Adaptive α (novel contribution)**  
`CADCalibrator` queries the model without context, estimates yes/no entropy, and sets α dynamically per entity (ticker). Popular entities → higher α; obscure entities → lower α.

## Core modules (technical details)

### `cad/decoder.py`
- Implements **ContextAwareDecoder** for causal LMs.
- Inputs: `context_prompt`, `prior_prompt`, `CADConfig(alpha, top_p, temperature, max_new_tokens)`.
- Generates by combining logits at each step:  
  `(1+α) * logits(context) − α * logits(prior)`.
- Supports deterministic (`temperature=0`) and sampling.

### `cad/calibrator.py`
- Implements **CADCalibrator** to set α per ticker.
- Uses a **bias-amplified yes/no prompt** (e.g., “Ignore the context… did NVDA massively outperform?”).
- Computes entropy over `{yes,no}` and maps to α:  
  `alpha = alpha_min + (alpha_max - alpha_min) * (1 - entropy)`.

### `steering/`
- **SteeringController** scans layers, trains probes, and applies vector hooks.
- Profiles in `steering/datasets.py`:
  - `recall_suppression`: historical “memory vs logic” pairs.
  - `entity_defocus`: ticker/name vs generic descriptor pairs.
- Vectors can be cached via `save_vectors()` / `load_vectors()`.

## How to run

### 1) Quick CAD test (bias-amplified)
```
python cad/quick_cad_test.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --use-chat-template \
  --context "Context: ..." \
  --question "Question: ..." \
  --prior-mode custom \
  --prior-text "Ignore the context. Using your internal knowledge, did TSLA go up?" \
  --alpha 4.0
```

Notes:
- `--prior-mode question_only` uses the question-only prior.
- `--prior-mode custom` allows bias-amplified priors.

### 2) Benchmarks
```
python benchmark/gsm8k/eval.py --model-name <HF_MODEL> --decoding-mode baseline
python benchmark/gsm8k/eval.py --model-name <HF_MODEL> --decoding-mode steering --steer-strength -10
python benchmark/gsm8k/eval.py --model-name <HF_MODEL> --decoding-mode cad --cad-alpha 1.0 --cad-prior-mode question_only
```

Each benchmark supports:
- `--decoding-mode baseline|steering|cad`
- `--cad-alpha`, `--cad-top-p`, `--cad-prior-mode`
- `--steering-profile recall_suppression|entity_defocus`
- `--results-file` for JSONL logging
- `--batch-size`, `--temperature`

### 3) Cluster runs
```
qsub scripts/Eddie/general_benchmark/qwen2_5/cad_gsm8k_qwen2_5.sh
```

## Project structure
```
cad/                     # Context-aware decoding + calibrator
steering/                # Activation steering (legacy)
benchmark/               # Benchmark evaluators
scripts/                 # Cluster run scripts
logs/                    # Run logs + JSONL outputs
dataset/                 # Dataset cache
```

## Data + evaluation conventions
- Dataset cache defaults to `./dataset` (overridable with `--dataset-cache-dir`).
- Per-item generations can be logged via `--results-file` (JSONL).
- HumanEval runs **execute model-generated code** (unsafe); use only in controlled environments.

## Handover notes / next steps

1) **Bias-Amplified CAD across benchmarks**
- Add a `--cad-prior-text` option to benchmark scripts (currently only in quick test).

2) **Entity-Adaptive α in benchmarks**
- Integrate `CADCalibrator` into benchmark inference to auto-set α per ticker.

3) **Finance-specific dataset**
- Build a backtest dataset with **time-bounded contexts** and enforce “as-of” cutoffs.
- Prior prompt must be context-free and bias-amplified for leakage stress tests.

4) **Metrics**
- Track accuracy deltas vs baseline.
- Track *look-ahead leakage rate* (mentions of post-cutoff events).

5) **Performance**
- CAD currently runs sequentially per example. Consider batched CAD or a multi-process CAD runner.

6) **Experiment hygiene**
- Always log per-item generations (`--results-file`).
- Keep vector caches separate for steering profiles.

## Dependencies
- Python 3.9+
- `torch`, `transformers`, `datasets`, `scikit-learn`, `huggingface_hub`

See `requirements.txt` or `environment.yml`.
