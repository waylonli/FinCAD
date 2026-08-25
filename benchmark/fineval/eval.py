"""
FinEval evaluation (baseline vs CAD).

Dataset: waylonli/fineval-processed (HuggingFace)
20 single-selection MC subsets covering CFA, credit risk, sentiment, ESG,
stock movement, etc.

Examples:
- Single subset baseline:
    python benchmark/fineval/eval.py \
        --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
        --subset MMLU-finance --max-samples 10 --decoding-mode baseline

- All subsets with CAD:
    python benchmark/fineval/eval.py \
        --model-name Qwen/Qwen2.5-14B-Instruct --use-chat-template \
        --subset all --decoding-mode cad --cad-alpha 1.0 \
        --cad-prior-mode question_only \
        --results-file logs/fineval_cad.jsonl
"""

from __future__ import annotations

import argparse
import json
import string
from itertools import islice
from pathlib import Path
from typing import IO, Iterable, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from cad import CADConfig, ContextAwareDecoder
from adapters import AdapterInitConfig, TransformersAdapter, seed_everything

LETTERS = list(string.ascii_uppercase) + [f"A{c}" for c in string.ascii_uppercase]

ALL_SUBSETS = [
    "CFA-Challenge",
    "CFA-Easy",
    "CRA-Bigdata",
    "CRA-CCF",
    "CRA-CCFraud",
    "CRA-LendingClub",
    "CRA-Polish",
    "CRA-ProtoSeguro",
    "CRA-Taiwan",
    "CRA-TravelInsurance",
    "FIQASA",
    "FOMC",
    "FPB",
    "Flare-Australian",
    "Flare-German",
    "MA",
    "MLESG",
    "MMLU-finance",
    "SM-ACL",
    "SM-CIKM",
]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(query: str, context: str, options: List[str]) -> str:
    rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    parts = [
        "Answer the multiple-choice question. Respond with the single letter of the correct option.\n",
    ]
    if context:
        parts.append(f"Context: {context}\n")
    parts.append(f"Question: {query}\n")
    parts.append(f"Options:\n{rendered_options}\n")
    parts.append("Answer:")
    return "\n".join(parts)


def format_prior_prompt(query: str, options: List[str], mode: str, neg_prompt_builder=None) -> str:
    if mode == "optimized" and neg_prompt_builder is not None:
        rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        output_format = (
            "Respond with the single letter of the correct option.\n\n"
            f"Options:\n{rendered_options}\n\n"
            "Answer:"
        )
        return neg_prompt_builder.build(task_prompt=output_format)
    if mode == "recall":
        rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        return (
            "Recall from your pretrained knowledge and select the most likely answer.\n\n"
            f"Options:\n{rendered_options}\n\n"
            "Answer:"
        )
    if mode == "question_only":
        rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        return (
            f"Question: {query}\n\n"
            f"Options:\n{rendered_options}\n\n"
            "Answer:"
        )
    # 'same' is handled by the caller (reuses the full prompt)
    return f"Question: {query}\nAnswer:"


# ---------------------------------------------------------------------------
# Answer helpers  (identical to mmlu_pro)
# ---------------------------------------------------------------------------

def normalize_gold(answer, num_options: int) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, int):
        if 0 <= answer < num_options:
            return LETTERS[answer]
        return None
    if isinstance(answer, str):
        ans = answer.strip().upper()
        if ans in LETTERS[:num_options]:
            return ans
    return None


def extract_prediction(text: str, num_options: int) -> Optional[str]:
    valid = set(LETTERS[:num_options])
    # Try two-char labels first (AA, AB, ...) if there are > 26 options
    if num_options > 26:
        import re
        two_char = re.findall(r"\b([A-Z]{2})\b", text)
        for match in reversed(two_char):
            if match in valid:
                return match
    # Fall back to single-letter match
    for ch in reversed(text):
        if ch.upper() in valid:
            return ch.upper()
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fineval(cache_dir: str) -> pd.DataFrame:
    """Load the full fineval-processed dataset from HuggingFace."""
    ds = load_dataset("waylonli/fineval-processed", split="train", cache_dir=cache_dir)
    return ds.to_pandas()


def load_subset(full_df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Filter the full dataframe to a single subset."""
    sub = full_df[full_df["subset"] == subset]
    if sub.empty:
        available = sorted(full_df["subset"].unique())
        raise ValueError(
            f"Subset '{subset}' not found. Available: {available}"
        )
    return sub


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    adapter: TransformersAdapter,
    cad_decoder: ContextAwareDecoder,
    prompts: List[str],
    prior_inputs: List[Tuple[str, List[str]]],
    decoding_mode: str,
    max_new_tokens: int,
    temperature: float,
    cad_config: CADConfig,
    cad_prior_mode: str,
    neg_prompt_builder=None,
):
    if decoding_mode == "cad":
        cad_config = CADConfig(
            alpha=cad_config.alpha,
            top_p=cad_config.top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        if cad_prior_mode == "same":
            prior_prompts = prompts
        else:
            prior_prompts = [
                format_prior_prompt(q, opts, cad_prior_mode, neg_prompt_builder=neg_prompt_builder)
                for q, opts in prior_inputs
            ]
        return cad_decoder.generate(prompts, prior_prompts, cad_config)
    # baseline
    inputs = adapter.tokenize(prompts)
    output_ids = adapter.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    input_len = inputs["input_ids"].shape[1]
    return [
        adapter.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
        for ids in output_ids
    ]


# ---------------------------------------------------------------------------
# Evaluate one subset
# ---------------------------------------------------------------------------

def evaluate_split(
    adapter: TransformersAdapter,
    cad_decoder: ContextAwareDecoder,
    samples: Iterable[dict],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    total_samples: int,
    decoding_mode: str,
    cad_config: CADConfig,
    cad_prior_mode: str,
    subset_name: str,
    results_file: Optional[IO[str]] = None,
    neg_prompt_builder=None,
) -> Tuple[int, int]:
    correct = 0
    total = 0
    batch_prompts: List[str] = []
    batch_prior_inputs: List[Tuple[str, List[str]]] = []
    golds: List[Optional[str]] = []
    option_counts: List[int] = []
    ids: List[int] = []
    batch_options: List[List[str]] = []

    def _flush():
        nonlocal correct, total
        generations = generate_batch(
            adapter, cad_decoder, batch_prompts, batch_prior_inputs,
            decoding_mode, max_new_tokens, temperature,
            cad_config, cad_prior_mode,
            neg_prompt_builder=neg_prompt_builder,
        )
        if isinstance(generations, str):
            generations = [generations]
        for gen, gold, num_opts, sid, prompt, opts in zip(
            generations, golds, option_counts, ids, batch_prompts, batch_options
        ):
            pred = extract_prediction(gen, num_opts)
            total += 1
            if gold is not None and pred == gold:
                correct += 1
            if results_file is not None:
                record = {
                    "id": sid,
                    "subset": subset_name,
                    "prompt": prompt,
                    "options": opts,
                    "gold": gold,
                    "prediction": pred,
                    "generation": gen,
                    "correct": gold is not None and pred == gold,
                    "decoding_mode": decoding_mode,
                }
                results_file.write(json.dumps(record) + "\n")

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc=f"Evaluating {subset_name}")):
        raw_choices = sample.get("choices")
        options = list(raw_choices) if raw_choices is not None and len(raw_choices) > 0 else []
        if not options:
            continue
        query = sample.get("query", "")
        context = sample.get("context", "")
        gold = normalize_gold(sample.get("answer_letter"), len(options))
        if gold is None:
            gold = normalize_gold(sample.get("answer_index"), len(options))

        golds.append(gold)
        option_counts.append(len(options))
        batch_prompts.append(format_prompt(query, context, options))
        batch_prior_inputs.append((query, options))
        ids.append(idx)
        batch_options.append(options)

        if len(batch_prompts) == batch_size:
            _flush()
            batch_prompts, golds, option_counts = [], [], []
            ids, batch_options, batch_prior_inputs = [], [], []
            if total % 20 == 0 and total > 0:
                print(f"  [{total} samples] Interim accuracy: {correct}/{total} = {correct / total:.2%}")

    if batch_prompts:
        _flush()

    return correct, total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinEval MC subsets: baseline vs CAD."
    )
    # FinEval-specific
    parser.add_argument("--subset", type=str, default="all",
                        help="Subset name or 'all' (default: all)")
    parser.add_argument("--dataset-cache-dir", type=str, default="./datasets",
                        help="Cache dir for dataset")
    # Standard
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default=None,
                        help="Optional cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true",
                        help="Apply the model's chat template to prompts")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples per subset (None for full)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation budget")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--results-file", type=str, default=None,
                        help="Optional JSONL path for per-item generations")
    parser.add_argument("--decoding-mode", type=str, default="baseline",
                        choices=["baseline", "cad"], help="Decoding mode")
    parser.add_argument("--cad-alpha", type=float, default=1.0,
                        help="CAD alpha for context-aware decoding")
    parser.add_argument("--cad-top-p", type=float, default=1.0,
                        help="Top-p filtering for CAD")
    parser.add_argument("--cad-prior-mode", type=str, default="same",
                        choices=["same", "question_only", "recall", "optimized"],
                        help="Prior prompt mode for CAD")
    parser.add_argument("--optimized-instruction", type=str, default=None,
                        help="Path to optimized instruction JSON (required when --cad-prior-mode=optimized)")
    parser.add_argument("--attn-implementation", type=str, default=None,
                        help="Attention implementation (e.g. flash_attention_2, sdpa). Auto-detected if omitted.")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile to the model (reduce-overhead mode)")
    parser.add_argument("--cache-file", type=str, default=None,
                        help="JSON cache for per-subset results (enables resume on interruption). "
                             "Auto-derived from --results-file if omitted.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    print(f"[Config] {args}")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            use_chat_template=args.use_chat_template,
            cache_dir=args.model_cache_dir,
            attn_implementation=args.attn_implementation,
            compile_model=args.compile,
        )
    )
    cad_decoder = ContextAwareDecoder(
        adapter.model,
        adapter.tokenizer,
        device=adapter.device,
        use_chat_template=args.use_chat_template,
    )

    if args.decoding_mode == "baseline":
        print("Running baseline (no steering).")
    elif args.decoding_mode == "cad":
        print("Running context-aware decoding (CAD).")

    # ------------------------------------------------------------------
    # Load dataset from HuggingFace
    # ------------------------------------------------------------------
    print(f"Loading fineval-processed from HuggingFace (cache: {args.dataset_cache_dir})...")
    full_df = load_fineval(args.dataset_cache_dir)
    print(f"Loaded {len(full_df)} rows across {full_df['subset'].nunique()} subsets.")

    # ------------------------------------------------------------------
    # Determine subsets to evaluate
    # ------------------------------------------------------------------
    if args.subset.lower() == "all":
        subsets = ALL_SUBSETS
    else:
        subsets = [args.subset]

    cad_config = CADConfig(
        alpha=args.cad_alpha,
        top_p=args.cad_top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    neg_prompt_builder = None
    if args.cad_prior_mode == "optimized":
        from cad.discovery import NegativePromptBuilder
        if args.optimized_instruction is None:
            print("ERROR: --optimized-instruction is required when --cad-prior-mode=optimized.")
            return
        neg_prompt_builder = NegativePromptBuilder.from_file(args.optimized_instruction)

    # ------------------------------------------------------------------
    # Cache setup (resume on interruption)
    # ------------------------------------------------------------------
    cache_path = args.cache_file
    if cache_path is None and args.results_file is not None:
        cache_path = str(Path(args.results_file).with_suffix(".cache.json"))

    cache: dict = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} completed subset(s) from {cache_path}")

    # Append to JSONL if resuming, otherwise overwrite
    results_mode = "a" if cache else "w"
    results_fh = open(args.results_file, results_mode) if args.results_file else None

    # ------------------------------------------------------------------
    # Evaluate each subset
    # ------------------------------------------------------------------
    summary: List[Tuple[str, int, int]] = []

    try:
        for subset_name in subsets:
            # Skip cached subsets
            if subset_name in cache:
                c, t = cache[subset_name]["correct"], cache[subset_name]["total"]
                acc = c / t if t else 0.0
                print(f"\n[CACHED] {subset_name}: {c}/{t} = {acc:.2%}")
                summary.append((subset_name, c, t))
                continue

            print(f"\n{'=' * 60}")
            print(f"Subset: {subset_name}")
            print(f"{'=' * 60}")

            df = load_subset(full_df, subset_name)
            total_samples = len(df) if args.max_samples is None else min(args.max_samples, len(df))
            samples = islice(df.to_dict("records"), total_samples)

            correct, total = evaluate_split(
                adapter,
                cad_decoder,
                samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
                total_samples=total_samples,
                decoding_mode=args.decoding_mode,
                cad_config=cad_config,
                cad_prior_mode=args.cad_prior_mode,
                subset_name=subset_name,
                results_file=results_fh,
                neg_prompt_builder=neg_prompt_builder,
            )
            acc = correct / total if total else 0.0
            summary.append((subset_name, correct, total))
            print(f"  {subset_name}: {correct}/{total} = {acc:.2%}")

            # Persist to cache after each subset
            cache[subset_name] = {"correct": correct, "total": total}
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)
            if results_fh is not None:
                results_fh.flush()
    finally:
        if results_fh is not None:
            results_fh.close()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    label = "CAD" if args.decoding_mode == "cad" else "Baseline"
    print(f"\n{'=' * 60}")
    print(f"FinEval {label} Results")
    print(f"{'=' * 60}")
    print(f"{'Subset':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 55)
    overall_correct = 0
    overall_total = 0
    for subset_name, c, t in summary:
        acc = c / t if t else 0.0
        print(f"{subset_name:<25} {c:>8} {t:>8} {acc:>10.2%}")
        overall_correct += c
        overall_total += t
    print("-" * 55)
    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print(f"{'OVERALL':<25} {overall_correct:>8} {overall_total:>8} {overall_acc:>10.2%}")


if __name__ == "__main__":
    main()
