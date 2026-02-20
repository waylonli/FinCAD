"""
HumanEval evaluation (baseline vs CAD).

Dataset: openai_humaneval (code generation). This runs generated code and test
snippets for scoring. Use with caution; executing model outputs can be unsafe.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from itertools import islice
from pathlib import Path
from typing import Iterable, Tuple, Optional, IO, List

from datasets import load_dataset
from tqdm import tqdm

from cad import CADConfig, ContextAwareDecoder
from adapters import AdapterInitConfig, TransformersAdapter


def format_prompt(problem: str) -> str:
    # HumanEval prompt already contains signature and docstring; just forward it.
    return problem + "\n"


def format_prior_prompt(problem: str, mode: str) -> str:
    if mode == "question_only":
        return problem + "\n"
    return format_prompt(problem)


def sanitize_generation(generation: str) -> str:
    """
    Extract a plausible code block from a free-form LLM generation.
    Priority:
    1) First fenced ```python ... ``` block (or ``` ... ```).
    2) Else take from first 'def ' onward.
    3) Fallback to raw text.
    """
    fences = re.findall(r"```(?:python)?\n(.*?)```", generation, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        candidate = fences[0]
    else:
        idx = generation.find("def ")
        candidate = generation[idx:] if idx != -1 else generation
    return candidate.strip()


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
    results_file: Optional[IO[str]] = None,
) -> Tuple[int, int]:
    passed = 0
    total = 0
    batch_prompts = []
    batch_prior = []
    batch_tests = []
    batch_entries = []
    ids = []

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc="Evaluating")):
        batch_prompts.append(format_prompt(sample["prompt"]))
        batch_prior.append(format_prior_prompt(sample["prompt"], cad_prior_mode))
        batch_tests.append(sample["test"])
        batch_entries.append(sample["entry_point"])
        ids.append(idx)

        if len(batch_prompts) == batch_size:
            generations = generate_batch(
                adapter,
                cad_decoder,
                batch_prompts,
                batch_prior,
                decoding_mode,
                max_new_tokens,
                temperature,
                cad_config,
            )
            if isinstance(generations, str):
                generations = [generations]
            for gen, test_code, entry, sid, prompt in zip(generations, batch_tests, batch_entries, ids, batch_prompts):
                candidate_code = sanitize_generation(gen)
                ok = run_candidate(candidate_code, test_code, entry)
                total += 1
                if ok:
                    passed += 1
                if results_file is not None:
                    record = {
                        "id": sid,
                        "prompt": prompt,
                        "generation": candidate_code,
                        "raw_generation": gen,
                        "correct": ok,
                        "decoding_mode": decoding_mode,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, batch_prior, batch_tests, batch_entries = [], [], [], []
            ids = []
            if total % 10 == 0:
                print(f"[{total} samples] Interim pass@1: {passed}/{total} = {passed/total:.2%}")

    if batch_prompts:
        generations = generate_batch(
            adapter,
            cad_decoder,
            batch_prompts,
            batch_prior,
            decoding_mode,
            max_new_tokens,
            temperature,
            cad_config,
        )
        if isinstance(generations, str):
            generations = [generations]
        for gen, test_code, entry, sid, prompt in zip(generations, batch_tests, batch_entries, ids, batch_prompts):
            candidate_code = sanitize_generation(gen)
            ok = run_candidate(candidate_code, test_code, entry)
            total += 1
            if ok:
                passed += 1
            if results_file is not None:
                record = {
                    "id": sid,
                    "prompt": prompt,
                    "generation": candidate_code,
                    "raw_generation": gen,
                    "correct": ok,
                    "decoding_mode": decoding_mode,
                }
                results_file.write(json.dumps(record) + "\n")

    return passed, total


def generate_batch(
    adapter: TransformersAdapter,
    cad_decoder: ContextAwareDecoder,
    prompts: List[str],
    priors: List[str],
    decoding_mode: str,
    max_new_tokens: int,
    temperature: float,
    cad_config: CADConfig,
):
    if decoding_mode == "cad":
        cad_config = CADConfig(
            alpha=cad_config.alpha,
            top_p=cad_config.top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return cad_decoder.generate(prompts, priors, cad_config)
    # baseline
    inputs = adapter.tokenize(prompts)
    output_ids = adapter.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    input_len = inputs["input_ids"].shape[1]
    return [
        adapter.tokenizer.decode(ids[input_len:], skip_special_tokens=True)
        for ids in output_ids
    ]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_ds_cache = repo_root / "dataset"

    parser = argparse.ArgumentParser(description="Evaluate HumanEval baseline vs CAD decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--results-file", type=str, default=None, help="Optional JSONL path for per-item generations")
    parser.add_argument("--decoding-mode", type=str, default="baseline", choices=["baseline", "cad"], help="Decoding mode")
    parser.add_argument("--cad-alpha", type=float, default=1.0, help="CAD alpha for context-aware decoding")
    parser.add_argument("--cad-top-p", type=float, default=1.0, help="Top-p filtering for CAD")
    parser.add_argument("--cad-prior-mode", type=str, default="same", choices=["same", "question_only"], help="Prior prompt mode for CAD")
    parser.add_argument("--attn-implementation", type=str, default=None, help="Attention implementation (e.g. flash_attention_2, sdpa). Auto-detected if omitted.")
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile to the model (reduce-overhead mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[Config] {args}")

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

    ds = load_dataset("openai_humaneval", cache_dir=args.dataset_cache_dir)
    split = ds["test"]
    total_samples = len(split) if args.max_samples is None else min(args.max_samples, len(split))
    samples = islice(split, total_samples)

    results_fh = open(args.results_file, "w") if args.results_file else None
    try:
        cad_config = CADConfig(
            alpha=args.cad_alpha,
            top_p=args.cad_top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        passed, total = evaluate_split(
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
            results_file=results_fh,
        )
    finally:
        if results_fh is not None:
            results_fh.close()
    pass_at_1 = passed / total if total else 0.0
    label = "CAD" if args.decoding_mode == "cad" else "Baseline"
    print(f"\n{label} pass@1 on HumanEval: {passed}/{total} = {pass_at_1:.2%}")


if __name__ == "__main__":
    # Warn about code execution risk.
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("default")
    print("Warning: This script executes model-generated code. Use in a safe environment.")
    main()
