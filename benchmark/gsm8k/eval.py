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
import json
import re
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Tuple, IO

from datasets import load_dataset
from tqdm import tqdm

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
    temperature: float,
    batch_size: int,
    total_samples: int,
    results_file: Optional[IO[str]] = None,
) -> Tuple[int, int]:
    correct = 0
    total = 0
    batch_prompts = []
    golds = []
    ids = []

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc="Evaluating")):
        question = sample["question"]
        gold_raw = sample["answer"]
        golds.append(extract_number(gold_raw))
        batch_prompts.append(format_prompt(question))
        ids.append(idx)

        if len(batch_prompts) == batch_size:
            generations = controller.generate(
                batch_prompts,
                strength=strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if isinstance(generations, str):
                generations = [generations]
            for gen, gold, sid, prompt in zip(generations, golds, ids, batch_prompts):
                pred = extract_number(gen)
                total += 1
                if gold is not None and pred is not None and gold.strip() == pred.strip():
                    correct += 1
                if results_file is not None:
                    record = {
                        "id": sid,
                        "prompt": prompt,
                        "gold": gold,
                        "prediction": pred,
                        "generation": gen,
                        "correct": gold is not None and pred is not None and gold.strip() == pred.strip(),
                        "steer_strength": strength,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, golds = [], []
            ids = []
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
        for gen, gold, sid, prompt in zip(generations, golds, ids, batch_prompts):
            pred = extract_number(gen)
            total += 1
            if gold is not None and pred is not None and gold.strip() == pred.strip():
                correct += 1
            if results_file is not None:
                record = {
                    "id": sid,
                    "prompt": prompt,
                    "gold": gold,
                    "prediction": pred,
                    "generation": gen,
                    "correct": gold is not None and pred is not None and gold.strip() == pred.strip(),
                    "steer_strength": strength,
                }
                results_file.write(json.dumps(record) + "\n")

    return correct, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate GSM8K baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples for a quick run (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for GSM8K")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget")
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
            scan_settings = ScanSettings(
                layer_step=2 if adapter.num_layers > 40 else 1,
            )
            controller.scan_layers(mem, gen, settings=scan_settings)
            controller.fit_steering_vectors(mem, gen)
            if args.vector_cache:
                Path(args.vector_cache).parent.mkdir(parents=True, exist_ok=True)
                controller.save_vectors(args.vector_cache)
    else:
        print("Running baseline (no steering).")

    ds = load_dataset("openai/gsm8k", "main", cache_dir=args.dataset_cache_dir)
    test_split = ds["test"]
    total_samples = len(test_split) if args.max_samples is None else min(args.max_samples, len(test_split))
    samples = islice(test_split, total_samples)

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
    print(f"\n{label} accuracy on GSM8K: {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
