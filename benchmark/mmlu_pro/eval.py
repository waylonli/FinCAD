"""
MMLU-Pro evaluation (baseline vs CAD).

Dataset: TIGER-Lab/MMLU-Pro
Fields assumed: question (str), options/choices (list[str]), answer (letter or index).

Examples:
- Baseline:
    python benchmark/mmlu_pro/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 100
- CAD:
    python benchmark/mmlu_pro/eval.py --model-name meta-llama/Llama-3.1-8B-Instruct --use-chat-template --max-samples 100 --decoding-mode cad
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

from cad import CADConfig, ContextAwareDecoder
from adapters import AdapterInitConfig, TransformersAdapter

LETTERS = list(string.ascii_uppercase)


def format_prompt(question: str, options: List[str]) -> str:
    rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return (
        "Answer the multiple-choice question. Respond with the single letter of the correct option.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{rendered_options}\n\n"
        "Answer:"
    )


def format_prior_prompt(question: str, options: List[str], mode: str, neg_prompt_builder=None) -> str:
    if mode == "optimized" and neg_prompt_builder is not None:
        rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        output_format = (
            "Respond with the single letter of the correct option.\n\n"
            f"Options:\n{rendered_options}\n\n"
            "Answer:"
        )
        return neg_prompt_builder.build(output_format_spec=output_format)
    if mode == "recall":
        rendered_options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        return (
            "Recall from your pretrained knowledge and select the most likely answer.\n\n"
            f"Options:\n{rendered_options}\n\n"
            "Answer:"
        )
    if mode == "question_only":
        return f"Question: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


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
    neg_prompt_builder=None,
) -> Tuple[int, int]:
    correct = 0
    total = 0
    batch_prompts = []
    batch_questions = []
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
        batch_questions.append(question)
        ids.append(idx)
        batch_options.append(options)

        if len(batch_prompts) == batch_size:
            generations = generate_batch(
                adapter,
                cad_decoder,
                batch_prompts,
                batch_questions,
                batch_options,
                decoding_mode,
                max_new_tokens,
                temperature,
                cad_config,
                cad_prior_mode,
                neg_prompt_builder=neg_prompt_builder,
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
                        "decoding_mode": decoding_mode,
                    }
                    results_file.write(json.dumps(record) + "\n")
            batch_prompts, golds, option_counts = [], [], []
            ids, batch_options, batch_questions = [], [], []
            if total % 20 == 0:
                print(f"[{total} samples] Interim accuracy: {correct}/{total} = {correct/total:.2%}")

    if batch_prompts:
        generations = generate_batch(
            adapter,
            cad_decoder,
            batch_prompts,
            batch_questions,
            batch_options,
            decoding_mode,
            max_new_tokens,
            temperature,
            cad_config,
            cad_prior_mode,
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
                    "decoding_mode": decoding_mode,
                }
                results_file.write(json.dumps(record) + "\n")

    return correct, total


def generate_batch(
    adapter: TransformersAdapter,
    cad_decoder: ContextAwareDecoder,
    prompts: List[str],
    questions: List[str],
    batch_options: List[List[str]],
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
            prior_prompts = [format_prior_prompt(q, opts, cad_prior_mode, neg_prompt_builder=neg_prompt_builder) for q, opts in zip(questions, batch_options)]
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

    parser = argparse.ArgumentParser(description="Evaluate MMLU-Pro baseline vs CAD decoding.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples (None for full)")
    parser.add_argument("--dataset-cache-dir", type=str, default=str(default_ds_cache), help="Cache dir for dataset")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation budget")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--results-file", type=str, default=None, help="Optional JSONL path for per-item generations")
    parser.add_argument("--decoding-mode", type=str, default="baseline", choices=["baseline", "cad"], help="Decoding mode")
    parser.add_argument("--cad-alpha", type=float, default=1.0, help="CAD alpha for context-aware decoding")
    parser.add_argument("--cad-top-p", type=float, default=1.0, help="Top-p filtering for CAD")
    parser.add_argument("--cad-prior-mode", type=str, default="same", choices=["same", "question_only", "recall", "optimized"], help="Prior prompt mode for CAD")
    parser.add_argument("--optimized-instruction", type=str, default=None, help="Path to optimized instruction JSON (required when --cad-prior-mode=optimized)")
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

    neg_prompt_builder = None
    if args.cad_prior_mode == "optimized":
        from cad.discovery import NegativePromptBuilder
        if args.optimized_instruction is None:
            print("ERROR: --optimized-instruction is required when --cad-prior-mode=optimized.")
            return
        neg_prompt_builder = NegativePromptBuilder.from_file(args.optimized_instruction)

    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=args.dataset_cache_dir)
    split = ds[args.split]
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
            neg_prompt_builder=neg_prompt_builder,
        )
    finally:
        if results_fh is not None:
            results_fh.close()
    acc = correct / total if total else 0.0
    label = "CAD" if args.decoding_mode == "cad" else "Baseline"
    print(f"\n{label} accuracy on MMLU-Pro ({args.split}): {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    main()
