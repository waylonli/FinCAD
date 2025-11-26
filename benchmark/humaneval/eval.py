"""
HumanEval evaluation (baseline vs steered).

Dataset: openai_humaneval (code generation). This runs generated code and test
snippets for scoring. Use with caution; executing model outputs can be unsafe.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from itertools import islice
from pathlib import Path
from typing import Iterable, Tuple

from datasets import load_dataset

from steering import (
    AdapterInitConfig,
    ScanSettings,
    SteeringController,
    TransformersAdapter,
    build_financial_contrast_pairs,
)


def format_prompt(problem: str) -> str:
    # HumanEval prompt already contains signature and docstring; just forward it.
    return problem + "\n"


def run_candidate(code: str, test_code: str, entry_point: str) -> bool:
    """
    Execute candidate code and dataset-provided tests. Not sandboxed; use carefully.
    Returns True if tests pass, False otherwise.
    """
    ns = {"__name__": "__main__"}
    try:
        exec(code, ns, ns)
        exec(test_code, ns, ns)
        return True
    except Exception:
        traceback.print_exc()
        return False


def evaluate_split(
    controller: SteeringController,
    samples: Iterable[dict],
    strength: float,
    max_new_tokens: int,
) -> Tuple[int, int]:
    passed = 0
    total = 0

    for idx, sample in enumerate(samples):
        prompt = format_prompt(sample["prompt"])
        generated = controller.generate(prompt, strength=strength, max_new_tokens=max_new_tokens)

        # Ensure we only keep the code portion (strip possible extra text).
        candidate_code = generated.strip()
        ok = run_candidate(candidate_code, sample["test"], sample["entry_point"])

        total += 1
        if ok:
            passed += 1

        if (idx + 1) % 10 == 0:
            print(f"[{idx + 1} samples] Interim pass@1: {passed}/{total} = {passed/total:.2%}")

    return passed, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate HumanEval baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=20, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
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

    ds = load_dataset("openai_humaneval", cache_dir=args.dataset_cache_dir)
    split = ds["test"]
    samples = split if args.max_samples is None else islice(split, args.max_samples)

    passed, total = evaluate_split(
        controller,
        samples,
        strength=args.steer_strength,
        max_new_tokens=args.max_new_tokens,
    )
    pass_at_1 = passed / total if total else 0.0
    label = "Steered" if args.steer_strength != 0.0 else "Baseline"
    print(f"\n{label} pass@1 on HumanEval: {passed}/{total} = {pass_at_1:.2%}")


if __name__ == "__main__":
    # Warn about code execution risk.
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("default")
    print("Warning: This script executes model-generated code. Use in a safe environment.")
    main()
