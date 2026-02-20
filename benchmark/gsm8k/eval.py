"""
GSM8K evaluation (baseline vs CAD).

Usage examples:
- Baseline:
    python benchmark/gsm8k/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 50
- CAD:
    python benchmark/gsm8k/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 50 --decoding-mode cad
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional, Tuple, IO, List

from datasets import load_dataset
from tqdm import tqdm

from cad import CADConfig, ContextAwareDecoder
from adapters import AdapterInitConfig, TransformersAdapter


def format_prompt(question: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step by step and give the final numeric answer after '####'.\n\n"
        f"Question: {question}\nAnswer:"
    )


def format_prior_prompt(question: str, mode: str) -> str:
    if mode == "recall":
        return "Recall from your pretrained knowledge. What is the numeric answer?\n\nAnswer:"
    if mode == "question_only":
        return f"Question: {question}\nAnswer:"
    return format_prompt(question)


def extract_number(text: str) -> Optional[str]:
    """Extract the final numeric answer (after '####' if present)."""
    segment = text.split("####")[-1] if "####" in text else text
    segment = segment.replace(",", " ")
    matches = re.findall(r"-?\d+(?:\.\d+)?", segment)
    if not matches:
        return None
    return matches[-1].lstrip("+")


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
    correct = 0
    total = 0
    batch_prompts = []
    batch_questions = []
    golds = []
    ids = []

    for idx, sample in enumerate(tqdm(samples, total=total_samples, desc="Evaluating")):
        question = sample["question"]
        gold_raw = sample["answer"]
        golds.append(extract_number(gold_raw))
        batch_prompts.append(format_prompt(question))
        batch_questions.append(question)
        ids.append(idx)

        if len(batch_prompts) == batch_size:
            generations = generate_batch(
                adapter,
                cad_decoder,
                batch_prompts,
                batch_questions,
                decoding_mode,
                max_new_tokens,
                temperature,
                cad_config,
                cad_prior_mode,
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
                        "decoding_mode": decoding_mode,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, golds = [], []
            batch_questions = []
            ids = []
            if total % 20 == 0:
                print(f"[{total} samples] Interim accuracy: {correct}/{total} = {correct/total:.2%}")

    if batch_prompts:
        generations = generate_batch(
            adapter,
            cad_decoder,
            batch_prompts,
            batch_questions,
            decoding_mode,
            max_new_tokens,
            temperature,
            cad_config,
            cad_prior_mode,
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
                    "decoding_mode": decoding_mode,
                }
                results_file.write(json.dumps(record) + "\n")

    return correct, total


def generate_batch(
    adapter: TransformersAdapter,
    cad_decoder: ContextAwareDecoder,
    prompts: List[str],
    questions: List[str],
    decoding_mode: str,
    max_new_tokens: int,
    temperature: float,
    cad_config: CADConfig,
    cad_prior_mode: str,
):
    if decoding_mode == "cad":
        cad_config = CADConfig(
            alpha=cad_config.alpha,
            top_p=cad_config.top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        prior_prompts = [format_prior_prompt(q, cad_prior_mode) for q in questions]
        return cad_decoder.generate(prompts, prior_prompts, cad_config)
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

    parser = argparse.ArgumentParser(description="Evaluate GSM8K baseline vs CAD decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of test samples for a quick run (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for GSM8K")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--results-file", type=str, default=None, help="Optional JSONL path for per-item generations")
    parser.add_argument("--decoding-mode", type=str, default="baseline", choices=["baseline", "cad"], help="Decoding mode")
    parser.add_argument("--cad-alpha", type=float, default=1.0, help="CAD alpha for context-aware decoding")
    parser.add_argument("--cad-top-p", type=float, default=1.0, help="Top-p filtering for CAD")
    parser.add_argument("--cad-prior-mode", type=str, default="same", choices=["same", "question_only", "recall"], help="Prior prompt mode for CAD")
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

    ds = load_dataset("openai/gsm8k", "main", cache_dir=args.dataset_cache_dir)
    test_split = ds["test"]
    total_samples = len(test_split) if args.max_samples is None else min(args.max_samples, len(test_split))
    samples = islice(test_split, total_samples)

    results_fh = open(args.results_file, "w") if args.results_file else None
    try:
        cad_config = CADConfig(
            alpha=args.cad_alpha,
            top_p=args.cad_top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
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
            results_file=results_fh,
        )
    finally:
        if results_fh is not None:
            results_fh.close()
    acc = correct / total if total else 0.0
    label = "CAD" if args.decoding_mode == "cad" else "Baseline"
    print(f"\n{label} accuracy on GSM8K: {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
