"""
GSM8K evaluation (baseline vs steered) using the steering controller.

Usage examples:
- Baseline:
    python benchmark/gsm8k/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 50
- Steered:
    python benchmark/gsm8k/eval.py --model-name zai-org/GLM-4-9B-0414 --use-chat-template --steer-strength -10 --max-samples 50
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
        "You are a careful math tutor. Solve the problem step by step and give the final numeric answer after '####'.\n\n"
        f"Question: {question}\nAnswer:"
    )


def extract_number(text: str) -> Optional[str]:
    """Extract the final numeric answer (after '####' if present)."""
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
        question = sample["question"]
        gold_raw = sample["answer"]
        gold = extract_number(gold_raw)

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

    parser = argparse.ArgumentParser(description="Evaluate GSM8K baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=50, help="Limit number of test samples for a quick run (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for GSM8K")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget")
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
        scan_settings = ScanSettings(
            layer_step=2 if adapter.num_layers > 40 else 1,
        )
        controller.scan_layers(mem, gen, settings=scan_settings)
        controller.fit_steering_vectors(mem, gen)
    else:
        print("Running baseline (no steering).")

    ds = load_dataset("openai/gsm8k", "main", cache_dir=args.dataset_cache_dir)
    test_split = ds["test"]
    samples = test_split if args.max_samples is None else islice(test_split, args.max_samples)

    correct, total = evaluate_split(
        controller,
        samples,
        strength=args.steer_strength,
        max_new_tokens=args.max_new_tokens,
    )
    acc = correct / total if total else 0.0
    label = "Steered" if args.steer_strength != 0.0 else "Baseline"
    print(f"\n{label} accuracy on GSM8K: {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
