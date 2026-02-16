#!/usr/bin/env python
"""
Quick CAD test: three-way comparison for a single prompt.

Generates output under three conditions:
  1. Baseline          – plain prompt, no mitigation
  2. Naive instruction – same prompt with "do not use knowledge after <date>"
  3. CAD               – logit-level bias subtraction via ContextAwareDecoder

This lets you see whether a simple prompt-level instruction is enough to
suppress look-ahead bias, compared to the CAD decoding-time approach.

Example:
  python cad/quick_cad_test.py \\
    --model-name Qwen/Qwen2.5-14B-Instruct \\
    --use-chat-template \\
    --context "Context: ..." \\
    --question "Question: ..." \\
    --prior-mode question_only \\
    --alpha 1.0 \\
    --cutoff-date 2023-01-01
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cad import CADConfig, ContextAwareDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick CAD prompt runner.")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--model-cache-dir", type=str, default="../pretrained_models", help="Cache dir for model weights")
    parser.add_argument("--use-chat-template", action="store_true", help="Apply the model's chat template to prompts")
    parser.add_argument("--context", type=str, required=True, help="Context string to condition on")
    parser.add_argument("--question", type=str, required=True, help="Question or instruction")
    parser.add_argument(
        "--prior-mode",
        type=str,
        default="question_only",
        choices=["same", "question_only", "custom"],
        help="Prior prompt mode for CAD",
    )
    parser.add_argument("--prior-text", type=str, default=None, help="Custom prior prompt text when --prior-mode=custom")
    parser.add_argument("--alpha", type=float, default=1.0, help="CAD alpha")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p filtering")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Knowledge cutoff date for the naive prompt instruction (e.g. '2023-01-01'). "
        "If not set, defaults to the current date.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[Config] {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    import os

    is_local = os.path.isdir(args.model_name)
    extra_kwargs = {} if is_local else {"cache_dir": args.model_cache_dir}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **extra_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        **extra_kwargs,
    )
    model.eval()

    context_prompt = f"{args.context}\n\n{args.question}"
    if args.prior_mode == "question_only":
        prior_prompt = args.question
    elif args.prior_mode == "custom":
        if not args.prior_text:
            raise ValueError("--prior-text is required when --prior-mode=custom")
        prior_prompt = args.prior_text
    else:
        prior_prompt = context_prompt

    decoder = ContextAwareDecoder(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_chat_template=args.use_chat_template,
    )
    config = CADConfig(
        alpha=args.alpha,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # -- Build the naive-instruction prompt (baseline + cutoff warning) --
    from datetime import date as _date

    cutoff = args.cutoff_date or str(_date.today())
    naive_instruction = (
        f"Important: Do not use any knowledge after {cutoff}. "
        "Base your answer solely on the information provided in the context above."
    )
    naive_prompt = f"{args.context}\n\n{naive_instruction}\n\n{args.question}"

    print("\n" + "=" * 60)
    print("PROMPTS")
    print("=" * 60)
    print("\n--- Context Prompt (Baseline & CAD) ---")
    print(context_prompt)
    print("\n--- Naive-Instruction Prompt ---")
    print(naive_prompt)
    print("\n--- Prior Prompt (CAD) ---")
    print(prior_prompt)

    # -- Helper to run standard generation --
    def _baseline_generate(prompt: str) -> str:
        enc = decoder._encode(prompt)
        ids = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # -- 1. Baseline: plain prompt, no mitigation --
    baseline_text = _baseline_generate(context_prompt)

    # -- 2. Naive instruction: prompt-level "do not use knowledge after X" --
    naive_text = _baseline_generate(naive_prompt)

    # -- 3. CAD: logit-level bias subtraction --
    cad_text = decoder.generate(context_prompt, prior_prompt, config)

    # -- Print results side by side --
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n=== 1. Baseline Output ===")
    print(baseline_text)

    print("\n=== 2. Naive Instruction Output ===")
    print(f"(injected: \"{naive_instruction}\")")
    print(naive_text)

    print("\n=== 3. CAD Output (alpha={:.2f}) ===".format(args.alpha))
    print(cad_text)


if __name__ == "__main__":
    main()
