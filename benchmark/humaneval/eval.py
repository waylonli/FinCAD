"""
HumanEval evaluation (baseline vs steered).

Dataset: openai_humaneval (code generation). This runs generated code and test
snippets for scoring. Use with caution; executing model outputs can be unsafe.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from itertools import islice
from pathlib import Path
from typing import Iterable, Tuple, Optional, IO

from datasets import load_dataset
from tqdm import tqdm

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
    temperature: float,
    batch_size: int,
    total_samples: int,
    results_file: Optional[IO[str]] = None,
) -> Tuple[int, int]:
    passed = 0
    total = 0
    batch_prompts = []
    batch_tests = []
    batch_entries = []
    ids = []

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc="Evaluating")):
        batch_prompts.append(format_prompt(sample["prompt"]))
        batch_tests.append(sample["test"])
        batch_entries.append(sample["entry_point"])
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
            for gen, test_code, entry, sid, prompt in zip(generations, batch_tests, batch_entries, ids, batch_prompts):
                candidate_code = gen.strip()
                ok = run_candidate(candidate_code, test_code, entry)
                total += 1
                if ok:
                    passed += 1
                if results_file is not None:
                    record = {
                        "id": sid,
                        "prompt": prompt,
                        "generation": candidate_code,
                        "correct": ok,
                        "steer_strength": strength,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, batch_tests, batch_entries = [], [], []
            ids = []
            if total % 10 == 0:
                print(f"[{total} samples] Interim pass@1: {passed}/{total} = {passed/total:.2%}")

    if batch_prompts:
        generations = controller.generate(
            batch_prompts,
            strength=strength,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if isinstance(generations, str):
            generations = [generations]
        for gen, test_code, entry, sid, prompt in zip(generations, batch_tests, batch_entries, ids, batch_prompts):
            candidate_code = gen.strip()
            ok = run_candidate(candidate_code, test_code, entry)
            total += 1
            if ok:
                passed += 1
            if results_file is not None:
                record = {
                    "id": sid,
                    "prompt": prompt,
                    "generation": candidate_code,
                    "correct": ok,
                    "steer_strength": strength,
                }
                results_file.write(json.dumps(record) + "\n")

    return passed, total


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate HumanEval baseline vs steered decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--steer-strength", type=float, default=0.0, help="Steering strength; 0 = baseline")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
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

    ds = load_dataset("openai_humaneval", cache_dir=args.dataset_cache_dir)
    split = ds["test"]
    total_samples = len(split) if args.max_samples is None else min(args.max_samples, len(split))
    samples = islice(split, total_samples)

    results_fh = open(args.results_file, "w") if args.results_file else None
    try:
        passed, total = evaluate_split(
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
