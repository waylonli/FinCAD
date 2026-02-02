#!/usr/bin/env python
"""
Quick CAD test: compare baseline vs CAD for a single prompt.

Example:
  python cad/quick_cad_test.py \\
    --model-name Qwen/Qwen2.5-14B-Instruct \\
    --use-chat-template \\
    --context "Context: ..." \\
    --question "Question: ..." \\
    --prior-mode question_only \\
    --alpha 1.0
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
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[Config] {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        cache_dir=args.model_cache_dir,
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

    print("\n=== Context Prompt ===")
    print(context_prompt)
    print("\n=== Prior Prompt ===")
    print(prior_prompt)

    encoded = decoder._encode(context_prompt)
    baseline_ids = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_len = encoded["input_ids"].shape[1]
    baseline_text = tokenizer.decode(baseline_ids[0][input_len:], skip_special_tokens=True)

    cad_text = decoder.generate(context_prompt, prior_prompt, config)

    print("\n=== Baseline Output ===")
    print(baseline_text)
    print("\n=== CAD Output ===")
    print(cad_text)


if __name__ == "__main__":
    main()
