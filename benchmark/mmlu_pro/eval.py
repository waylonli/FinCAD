"""
MMLU-Pro evaluation (baseline vs steered).

Dataset: TIGER-Lab/MMLU-Pro
Fields assumed: question (str), options/choices (list[str]), answer (letter or index).

Examples:
- Baseline:
    python benchmark/mmlu_pro/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 100
- Steered:
    python benchmark/mmlu_pro/eval.py --model-name zai-org/GLM-4-9B-0414 --use-chat-template --steer-strength -10 --max-samples 100
"""

from __future__ import annotations

import argparse
import string
import json
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, IO

from datasets import load_dataset
from tqdm import tqdm

from steering import (
    AdapterInitConfig,
    ScanSettings,
    SteeringController,
    TransformersAdapter,
    build_financial_contrast_pairs,
)

LETTERS = list(string.ascii_uppercase)


def format_prompt(question: str, options: List[str]) -> str:
    rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        "Answer the multiple-choice question. Respond with the single letter of the correct option.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{rendered_options}\n\n"
        "Answer:"
    )


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
    # Take the last letter found that matches an option label.
    for ch in reversed(text):
        if ch.upper() in LETTERS[:num_options]:
            return ch.upper()
    return None


def evaluate_split(
    controller: SteeringController,
    samples: Iterable[dict],
    strength: float,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    total_samples: int,
    results_file: Optional[IO[str]] = None,
) -> Tuple[int, int]:
    correct = 0
    total = 0
    batch_prompts = []
    golds = []
    option_counts = []
    ids = []
    batch_options = []

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc="Evaluating")):
        options = sample.get("options") or sample.get("choices") or []
        if not options:
            continue
        question = sample.get("question", "")
        golds.append(normalize_gold(sample.get("answer"), len(options)))
        option_counts.append(len(options))
        batch_prompts.append(format_prompt(question, options))
        ids.append(idx)
        batch_options.append(options)

        if len(batch_prompts) == batch_size:
            generations = controller.generate(
                batch_prompts,
                strength=strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if isinstance(generations, str):
                generations = [generations]
            for gen, gold, num_opts, sid, prompt, opts in zip(generations, golds, option_counts, ids, batch_prompts, batch_options):
                pred = extract_prediction(gen, num_opts)
                total += 1
                if gold is not None and pred == gold:
                    correct += 1
                if results_file is not None:
                    record = {
                        "id": sid,
                        "prompt": prompt,
                        "options": opts,
                        "gold": gold,
                        "prediction": pred,
                        "generation": gen,
                        "correct": gold is not None and pred == gold,
                        "steer_strength": strength,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, golds, option_counts = [], [], []
            ids, batch_options = [], []
            if total % 20 == 0:
                print(f"[{total} samples] Interim accuracy: {correct}/{total} = {correct/total:.2%}")

    if batch_prompts:
        generations = controller.generate(
            batch_prompts,
            strength=strength,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if isinstance(generations, str):
            generations = [generations]
        for gen, gold, num_opts, sid, prompt, opts in zip(generations, golds, option_counts, ids, batch_prompts, batch_options):
            pred = extract_prediction(gen, num_opts)
            total += 1
            if gold is not None and pred == gold:
                correct += 1
            if results_file is not None:
                record = {
                    "id": sid,
                    "prompt": prompt,
                    "options": opts,
                    "gold": gold,
                    "prediction": pred,
                    "generation": gen,
                    "correct": gold is not None and pred == gold,
                    "steer_strength": strength,
                }
                results_file.write(json.dumps(record) + "\n")

    return correct, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate MMLU-Pro baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation budget")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--vector-cache", type=str, default=None, help="Path to load/save steering vectors")
    parser.add_argument("--results-file", type=str, default=None, help="Optional JSONL path for per-item generations")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[Config] {args}")

    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            use_chat_template=args.use_chat_template,
            cache_dir=args.model_cache_dir,
        )
    )
    controller = SteeringController(adapter)

    if args.steer_strength != 0.0:
        if args.vector_cache and Path(args.vector_cache).exists():
            print(f"Loading steering vectors from {args.vector_cache}")
            controller.load_vectors(args.vector_cache)
        else:
            mem, gen = build_financial_contrast_pairs()
            scan_settings = ScanSettings(layer_step=2 if adapter.num_layers > 40 else 1)
            controller.scan_layers(mem, gen, settings=scan_settings)
            controller.fit_steering_vectors(mem, gen)
            if args.vector_cache:
                Path(args.vector_cache).parent.mkdir(parents=True, exist_ok=True)
                controller.save_vectors(args.vector_cache)
    else:
        print("Running baseline (no steering).")

    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=args.dataset_cache_dir)
    split = ds[args.split]
    total_samples = len(split) if args.max_samples is None else min(args.max_samples, len(split))
    samples = islice(split, total_samples)

    results_fh = open(args.results_file, "w") if args.results_file else None
    try:
        correct, total = evaluate_split(
            controller,
            samples,
            strength=args.steer_strength,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            total_samples=total_samples,
            results_file=results_fh,
        )
    finally:
        if results_fh is not None:
            results_fh.close()
    acc = correct / total if total else 0.0
    label = "Steered" if args.steer_strength != 0.0 else "Baseline"
    print(f"\n{label} accuracy on MMLU-Pro ({args.split}): {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
