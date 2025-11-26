"""
Competition Math evaluation (baseline vs steered).

Dataset: qwedsacf/competition_math
Assumed fields: problem/question (str), answer/solution (numeric-ish string).
Scoring uses numeric extraction (last number) similar to GSM8K.
"""

from __future__ import annotations

import argparse
import re
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Tuple

from datasets import load_dataset

from steering import (
    AdapterInitConfig,
    ScanSettings,
    SteeringController,
    TransformersAdapter,
    build_financial_contrast_pairs,
)


def format_prompt(question: str) -> str:
    return (
        "Solve the competition math problem step by step. Provide the final numeric answer after '####'.\n\n"
        f"Problem: {question}\nAnswer:"
    )


def extract_number(text: str) -> Optional[str]:
    segment = text.split("####")[-1] if "####" in text else text
    segment = segment.replace(",", " ")
    matches = re.findall(r"-?\d+(?:\.\d+)?", segment)
    if not matches:
        return None
    return matches[-1].lstrip("+")


def evaluate_split(
    controller: SteeringController,
    samples: Iterable[dict],
    strength: float,
    max_new_tokens: int,
) -> Tuple[int, int]:
    correct = 0
    total = 0

    for idx, sample in enumerate(samples):
        question = sample.get("problem") or sample.get("question") or ""
        gold_raw = sample.get("answer") or sample.get("solution") or ""
        gold = extract_number(str(gold_raw))

        prompt = format_prompt(question)
        generated = controller.generate(prompt, strength=strength, max_new_tokens=max_new_tokens)
        pred = extract_number(generated)

        total += 1
        if gold is not None and pred is not None and gold.strip() == pred.strip():
            correct += 1

        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1} samples] Interim accuracy: {correct}/{total} = {correct/total:.2%}")

    return correct, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate competition math baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=50, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
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

    ds = load_dataset("qwedsacf/competition_math", cache_dir=args.dataset_cache_dir)
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
    print(f"\n{label} accuracy on competition_math ({args.split}): {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
