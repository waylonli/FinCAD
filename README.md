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

**3) General benchmarks (safety check)**
- Folder: `benchmark/gsm8k/`, `benchmark/mmlu_pro/`, `benchmark/competition_math/`, `benchmark/humaneval/`
- Benchmarks implemented: GSM8K, MMLU-Pro, MATH-500, HumanEval.
- Each script supports baseline, steering, and CAD via `--decoding-mode`.
- Purpose: verify that CAD does not degrade general reasoning ability.

**4) Financial backtesting (honesty evaluation)**
- Folder: `benchmark/backtest/`
- Two complementary approaches (see [Financial Backtesting](#financial-backtesting) below).
- Purpose: verify that CAD suppresses look-ahead bias by comparing `Returns_CAD < Returns_Baseline`.

**5) Cluster scripts**
- Folder: `scripts/`
- Run scripts under:
  - `scripts/Eddie/general_benchmark/qwen2_5/`
  - `scripts/lightspeed/general_benchmark/qwen2_5/`
  - `scripts/lightspeed/backtest/qwen2_5_14b/`
  - `scripts/lightspeed/backtest/phi4_14b/`

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

### 4) Cluster runs
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
  lightspeed/backtest/                  #   Cluster scripts (qwen2_5_14b, phi4_14b)
  precompute_adjusted_open.py           #   Offline price adjustment + validation
  validate_adjusted_open.py             #   Detailed gap/continuity checks
dataset/
  backtest-data/price/price_data.csv    #   S&P 500 price data (2000-2024, with adjusted_open)
logs/                                   # Run logs + JSONL outputs
```

## Data + evaluation conventions
- Dataset cache defaults to `./dataset` (overridable with `--dataset-cache-dir`).
- Per-item generations can be logged via `--results-file` (JSONL).
- HumanEval runs **execute model-generated code** (unsafe); use only in controlled environments.
