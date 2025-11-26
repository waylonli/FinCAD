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
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset

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
) -> Tuple[int, int]:
    correct = 0
    total = 0

    for idx, sample in enumerate(samples):
        options = sample.get("options") or sample.get("choices") or []
        if not options:
            continue
        question = sample.get("question", "")
        gold = normalize_gold(sample.get("answer"), len(options))
        prompt = format_prompt(question, options)

        generated = controller.generate(prompt, strength=strength, max_new_tokens=max_new_tokens)
        pred = extract_prediction(generated, len(options))

        total += 1
        if gold is not None and pred == gold:
            correct += 1

        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1} samples] Interim accuracy: {correct}/{total} = {correct/total:.2%}")

    return correct, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate MMLU-Pro baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=100, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation budget")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()

    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            use_chat_template=args.use_chat_template,
            cache_dir=args.model_cache_dir,
        )
    )
    controller = SteeringController(adapter)

    if args.steer_strength != 0.0:
        mem, gen = build_financial_contrast_pairs()
        scan_settings = ScanSettings(layer_step=2 if adapter.num_layers > 40 else 1)
        controller.scan_layers(mem, gen, settings=scan_settings)
        controller.fit_steering_vectors(mem, gen)
    else:
        print("Running baseline (no steering).")

    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=args.dataset_cache_dir)
    split = ds[args.split]
    samples = split if args.max_samples is None else islice(split, args.max_samples)

    correct, total = evaluate_split(
        controller,
        samples,
        strength=args.steer_strength,
        max_new_tokens=args.max_new_tokens,
    )
    acc = correct / total if total else 0.0
    label = "Steered" if args.steer_strength != 0.0 else "Baseline"
    print(f"\n{label} accuracy on MMLU-Pro ({args.split}): {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
