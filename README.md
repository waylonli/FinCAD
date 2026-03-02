# look-ahead-bias

This repository mitigates **look-ahead bias** in LLM-based financial backtesting—where models use pretrained knowledge of future outcomes (e.g., known winners or crises) instead of relying strictly on information available at the historical time. The codebase provides **two inference-time control methods**, benchmarking pipelines, and cluster scripts for reproducible experiments. This README is written as a **handover document** for a new maintainer.

## Project objective

Build practical, inference-time mechanisms that:
- **Suppress pretrained “future knowledge”** leakage in finance tasks.
- Preserve general benchmark performance as a sanity check.
- Provide **reproducible experiments** with logs and per-item outputs.

## What’s in the repo

**1) Context-Aware Decoding (CAD)**
- Module: `cad/`
- Goal: force generation to depend on provided context by subtracting a biased “prior” from the context-conditioned logits.
- Supports **Bias-Amplified CAD** and **Entity-Adaptive α** via `CADCalibrator`.
- Prior modes: `bias_amplified`, `no_context`, `recall`, `optimized` (via `--cad-prior-mode`).

**1b) Adversarial Bias Discovery (DSPy optimization)**
- Module: `cad/discovery/`
- Goal: automatically optimize the memory-activation instruction in the negative prompt using DSPy MIPROv2/COPRO.
- Calibrates on historical price data (known future directions) to find the instruction that maximally activates parametric memory.
- See [Adversarial Bias Discovery](#adversarial-bias-discovery-via-dspy-caddiscovery) for full details.

**2) General benchmarks (safety check)**
- Folder: `benchmark/gsm8k/`, `benchmark/mmlu_pro/`, `benchmark/competition_math/`, `benchmark/humaneval/`
- Benchmarks implemented: GSM8K, MMLU-Pro, MATH-500, HumanEval.
- Each script supports baseline, steering, and CAD via `--decoding-mode`.
- Purpose: verify that CAD does not degrade general reasoning ability.

**3) Financial backtesting (honesty evaluation)**
- Folder: `benchmark/backtest/`
- Two complementary approaches (see [Financial Backtesting](#financial-backtesting) below).
- Purpose: verify that CAD suppresses look-ahead bias by comparing `Returns_CAD < Returns_Baseline`.

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

## Adversarial Bias Discovery via DSPy (`cad/discovery/`)

The negative prompt in CAD has the structure: `[Memory Activation Instruction T*] + [Task Instruction F_task]`. T\* is **model-specific but task-agnostic** — it maximally activates the model's parametric memory regardless of whether the downstream task is trading, scoring filings, MCQ, or math. F\_task is **task-specific** — it includes the task framing and output format, copied from the context prompt to preserve logit alignment for the CAD subtraction.

This module uses **discrete prompt optimization** (DSPy MIPROv2 / COPRO) to find the optimal memory activation instruction for a given model. The instruction is optimized once per model, then reused across all tasks.

### How it works

**Stage 1: Build calibration dataset from price data**

The calibration dataset converts historical S&P 500 prices into (ticker, date, direction) triples that probe whether the model "knows" future outcomes:

1. Load `price_data.csv` (columns: `date`, `symbol`, `adjusted_close`).
2. For each ticker, resample dates at quarter-end frequency (`QE`) within the date range (default: 2005–2015).
3. At each sampled date, compute the **63-trading-day forward return**: `(price[t+63] - price[t]) / price[t]`.
4. Filter out flat moves where `|return| < 5%` — these are ambiguous and uninformative.
5. Label direction: `return > 0` → `"up"`, `return < 0` → `"down"`.
6. Balance up/down classes (take min of each), cap at `max_examples` (default: 200).

This yields ~200 balanced examples where the ground-truth direction is unambiguous.

**Stage 2: DSPy prompt optimization**

Each calibration example is converted to a `dspy.Example(task=..., direction=direction)` where the `task` field contains the entity, date, and a fixed calibration instruction (`F_calib`): *"Predict whether the stock price went up or down after this date."* Examples are split 80/20 into train/val sets.

The DSPy `MemoryProbe` signature has a single input field `task` (generic description: "Task instruction") and output field `direction` (generic description: "Your prediction"). MIPROv2 only optimizes the signature's docstring (the memory-activation instruction T\*). To ensure T\* stays **task-agnostic**, the proposer's data-awareness and few-shot-awareness are disabled (`data_aware_proposer=False, fewshot_aware_proposer=False`), so the proposer never sees the actual training data values containing stock tickers or calibration task text. It only sees the generic field names and the current seed instruction. The evaluation metric, however, runs the full input through the model, so candidates are scored on actual memory-activation performance.

The metric `bias_activation_score` returns `1.0` if the model's predicted direction matches the ground-truth future return, `0.0` otherwise. A higher score means the instruction more effectively activates memorized knowledge — exactly the bias we want the prior prompt to elicit so CAD can subtract it.

MIPROv2 explores `num_candidates` instruction variants over `num_trials` evaluation rounds, selecting the instruction that maximizes val-set accuracy.

**Stage 3: Build negative prompts at inference time**

`NegativePromptBuilder` loads the saved instruction and constructs the full prior prompt:

```
[Optimized Memory Activation Instruction T*]  ← model-specific, task-agnostic
                                               (with {entity} and {date} placeholders filled)

[Task Instruction F_task]                     ← task-specific, copied from context prompt
                                               (e.g., "Analyse this stock and return JSON
                                                with signal, confidence, reasoning")
```

F\_task must match the context prompt's task framing and output format to keep the logit distributions aligned for `(1+α) * logits(context) − α * logits(prior)`.

### Module structure

| File | Purpose |
|------|---------|
| `config.py` | Dataclasses: `CalibrationExample`, `CalibrationDatasetConfig`, `DiscoveryConfig`, `OptimizedInstruction` |
| `calibration_data.py` | `build_calibration_dataset()` — price CSV → labelled examples; `to_dspy_examples()` — convert to DSPy format |
| `signatures.py` | `MemoryProbe(task) → direction` — DSPy Signature whose docstring (T\*) is optimized |
| `modules.py` | `MemoryProbeModule` — DSPy Module wrapping `Predict(MemoryProbe)` with up/down normalization |
| `metrics.py` | `bias_activation_score(example, prediction)` — 1.0 if direction matches, 0.0 otherwise |
| `optimizer.py` | `run_optimization(cfg)` — full pipeline: build data → split → configure LM → run MIPROv2/COPRO → evaluate → save |
| `builder.py` | `NegativePromptBuilder` — loads optimized instruction, substitutes `{entity}`/`{date}`, appends task instruction F\_task |
| `registry.py` | `save_instruction()` / `load_instruction()` — JSON persistence at `results/discovery/{model_slug}.json` |
| `__main__.py` | CLI entry point: `python -m cad.discovery` |

### CLI usage

```bash
# Option 1: In-process with HuggingFace transformers (no server needed)
python -m cad.discovery \
  --model-name /path/to/model \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --optimizer MIPROv2 \
  --num-candidates 10 --num-trials 30 \
  --forward-days 63 --max-examples 200 --min-abs-return 0.05 \
  --date-range-start 2005-01-01 --date-range-end 2015-01-01 \
  --output-dir results/discovery

# Option 2: Via a running vLLM / TGI server
python -m cad.discovery \
  --model-name /path/to/model \
  --server-url http://localhost:8234/v1 \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --num-trials 30 --num-candidates 10
```

The cluster script `scripts/optimize_prior.sh` runs optimization for both phi-4 (14B) and gemma-3-12b-it sequentially. Set `BACKEND=vllm` to auto-start a vLLM server per model, or `BACKEND=transformers` (default) to load in-process.

### Using the optimized instruction in benchmarks

All benchmark scripts support `--cad-prior-mode optimized --optimized-instruction <path>`:

```bash
# Single-stock backtest with optimized prior
python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \
  --rebalance-freq B --decoding-mode cad --cad-alpha 1.0 \
  --cad-prior-mode optimized \
  --optimized-instruction results/discovery/phi-4.json
```

The `NegativePromptBuilder` is constructed from the JSON file and threaded through each benchmark's prompt-building function. The builder substitutes entity/date placeholders and appends the task-specific instruction F\_task (full task framing + output format: JSON signal for trading, single-letter for MCQ, numeric for math, etc.).

### DSPy API notes

When using `dspy.MIPROv2`, set `auto=None` explicitly to use manual `num_candidates` and `num_trials`. The default `auto="light"` overrides any explicit values and raises a `ValueError`. `num_candidates` is passed to the constructor; `num_trials` is passed to `.compile()`.

## Financial Backtesting

The project includes two backtesting pipelines under `benchmark/backtest/`, each designed to measure whether CAD reduces look-ahead bias in LLM-generated financial decisions. Both pipelines use open-source models (Qwen2.5-14B-Instruct, Phi-4) running locally via HuggingFace Transformers.

### Method 1: Q-Score Portfolio Backtest (`benchmark/backtest/q_scores_eval/`)

**Approach:** The LLM scores SEC 10-Q filings across five risk-factor categories (Industry Position, Business Model, Financial Strength, Management Quality, ESG). Scores are aggregated into a quality signal, and a long-only portfolio is constructed by ranking stocks within the S&P 500 universe.

**Modules:**
- `prompts.py` — Five category prompts (from abrdn-risk-factor-eval) + CAD prior templates (no-context and bias-amplified).
- `data.py` — Price/filing/factor data loading; handles corrupted PDFs gracefully.
- `scorer.py` — `HFFilingScorer` generates quality scores using `ContextAwareDecoder`; supports score caching.
- `backtest.py` — Extracted backtest engine: signal panel, weight construction, daily NAV simulation, Fama-French factor regression.
- `eval.py` — CLI entry point. Supports `--score-mode precomputed|on_demand`.

**Usage:**
```bash
python -m benchmark.backtest.q_scores_eval.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --score-mode on_demand --decoding-mode cad --cad-alpha 1.0 \
  --symbols mag7 --start-year 2018 --end-year 2020
```

### Method 2: AI Hedge Fund Single-Stock Backtest (`benchmark/backtest/ai_hedge_fund/`)

**Approach:** The LLM acts as a quantitative analyst for a single stock. At each rebalance date it receives a structured financial summary (price-derived metrics only — no news or filing text) and returns a buy/sell/hold signal as JSON. This is much faster than Method 1 since it avoids scoring hundreds of filings.

**Modules:**
- `agent.py` — `TradingAgent` with CAD support; `build_financial_summary()` computes returns, moving averages, volatility, 52-week range from price data.
- `eval.py` — CLI entry point with `SimplePortfolio`, metrics, and buy-and-hold benchmark.

**Usage:**
```bash
# Baseline
python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \
  --rebalance-freq B --decoding-mode baseline

# CAD with bias-amplified prior
python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \
  --rebalance-freq B --decoding-mode cad --cad-alpha 1.0 --cad-prior-mode bias_amplified
```

Cluster scripts accept the ticker as the first argument (default: NVDA):
```bash
qsub scripts/lightspeed/backtest/qwen2_5_14b/baseline_single_stock_qwen2_5_14b.sh AAPL
qsub scripts/lightspeed/backtest/qwen2_5_14b/cad_single_stock_qwen2_5_14b.sh AAPL
```

Supported rebalance frequencies: `B` (business daily), `W` (weekly), `M` (monthly), `Q` (quarterly).

### Backtesting methodology

The backtesting engine follows standard quantitative finance conventions to avoid introducing bias at the simulation level:

**Price data:**
- All prices are split- and dividend-adjusted for consistency. The CSV contains `adjusted_close` (from provider) and `adjusted_open` (precomputed offline).
- `adjusted_open` is derived as: `adjusted_open = open × (adjusted_close / close)`, where the ratio is the cumulative split/dividend adjustment factor.
- Precomputation script: `scripts/precompute_adjusted_open.py` (includes 8 validation checks).

**Execution timing (no intraday look-ahead):**
- On each rebalance date, the agent sees data up to the **previous day's close** only (`date < rebalance_date`).
- Trades are executed at the **next trading day's adjusted open** price.
- Daily portfolio values are marked-to-market using `adjusted_close`.
- This ensures the agent never observes the price at which it trades.

**Commission model:**
- Moomoo US standard equity fees: **$0.0049/share**, minimum **$0.99/order**.
- Applied to both buys and sells; deducted from cash on execution.
- Commission is included in cost basis and tracked cumulatively.
- The buy-and-hold benchmark also pays commission on the initial purchase.

**Risk-free rate:**
- Historical average of **3% annualised** (`RF_ANNUAL = 0.03`).
- Used in Sharpe and Sortino ratio calculations.

**Performance metrics:**
- Total return, CAGR, annualised volatility, Sharpe ratio, Sortino ratio, maximum drawdown, ending portfolio value.
- Compared against a buy-and-hold benchmark (same entry timing and commission).

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

### 2) General benchmarks
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

### 3) Financial backtest (single-stock)
```bash
# Baseline — daily rebalance on NVDA
python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \
  --rebalance-freq B --decoding-mode baseline \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --results-file results/backtest/single_NVDA_baseline.jsonl \
  --summary-file results/backtest/single_NVDA_baseline_summary.json

# CAD — same setup with bias-amplified prior
python -m benchmark.backtest.ai_hedge_fund.eval \
  --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
  --ticker NVDA --start-date 2018-01-01 --end-date 2020-01-01 \
  --rebalance-freq B --decoding-mode cad --cad-alpha 1.0 --cad-prior-mode bias_amplified \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --results-file results/backtest/single_NVDA_cad.jsonl \
  --summary-file results/backtest/single_NVDA_cad_summary.json
```

### 4) Optimize prior (DSPy discovery)
```bash
# Run locally — in-process with transformers (no server needed)
python -m cad.discovery \
  --model-name /path/to/model \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --num-trials 30 --num-candidates 10

# Run locally — via a running vLLM server
python -m cad.discovery \
  --model-name /path/to/model \
  --server-url http://localhost:8234/v1 \
  --price-csv dataset/backtest-data/price/price_data.csv \
  --num-trials 30 --num-candidates 10

# Cluster: optimize for phi-4 and gemma-3 sequentially
# Set BACKEND=vllm to auto-start vLLM, or BACKEND=transformers (default)
qsub scripts/optimize_prior.sh
```

Output: `results/discovery/{model_slug}.json` containing the optimized instruction, val score, and metadata.

### 5) Cluster runs
```bash
# General benchmark
qsub scripts/Eddie/general_benchmark/qwen2_5/cad_gsm8k_qwen2_5.sh

# Single-stock backtest (pass ticker as first arg, default NVDA)
qsub scripts/lightspeed/backtest/qwen2_5_14b/baseline_single_stock_qwen2_5_14b.sh NVDA
qsub scripts/lightspeed/backtest/qwen2_5_14b/cad_single_stock_qwen2_5_14b.sh NVDA
```

## Project structure
```
cad/                                    # Context-aware decoding + calibrator
  decoder.py                            #   ContextAwareDecoder (logit-level CAD)
  calibrator.py                         #   CADCalibrator (entity-adaptive α)
  quick_cad_test.py                     #   Three-way comparison: baseline / naive / CAD
  discovery/                            #   Adversarial bias discovery via DSPy
    __main__.py                         #     CLI: python -m cad.discovery
    config.py                           #     Dataclasses (configs, examples, instructions)
    calibration_data.py                 #     Price CSV → labelled (ticker, date, direction)
    signatures.py                       #     MemoryProbe DSPy Signature
    modules.py                          #     MemoryProbeModule (Predict wrapper)
    metrics.py                          #     bias_activation_score metric
    optimizer.py                        #     Full optimization pipeline
    builder.py                          #     NegativePromptBuilder (instruction + format)
    registry.py                         #     Save/load optimized instructions as JSON
steering/                               # Activation steering (legacy)
benchmark/
  gsm8k/ mmlu_pro/ competition_math/    #   General benchmarks (safety check)
  humaneval/
  backtest/
    q_scores_eval/                      #   Method 1: Q-score portfolio backtest
      prompts.py data.py scorer.py      #     Filing scoring with CAD
      backtest.py eval.py               #     Portfolio construction + metrics
    ai_hedge_fund/                      #   Method 2: Single-stock backtest
      agent.py                          #     TradingAgent (buy/sell/hold signals)
      eval.py                           #     Backtest loop + metrics + commission
scripts/
  optimize_prior.sh                     #   DSPy optimization for phi-4 + gemma-3 (cluster)
  lightspeed/backtest/                  #   Cluster scripts (qwen2_5_14b, phi4_14b)
  precompute_adjusted_open.py           #   Offline price adjustment + validation
  validate_adjusted_open.py             #   Detailed gap/continuity checks
dataset/
  backtest-data/price/price_data.csv    #   S&P 500 price data (2000-2024, with adjusted_open)
results/
  discovery/                            #   Optimized instruction JSONs (per model)
logs/                                   # Run logs + JSONL outputs
```

## Data + evaluation conventions
- Dataset cache defaults to `./dataset` (overridable with `--dataset-cache-dir`).
- Per-item generations can be logged via `--results-file` (JSONL).
- HumanEval runs **execute model-generated code** (unsafe); use only in controlled environments.
