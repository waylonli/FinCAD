"""Command-line interface for applying a saved FinCAD discovery profile.

This module is intentionally a thin wrapper around the released decoder,
calibrator, and negative-prompt builder. It does not alter their numerical
or prompting behavior.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from adapters import AdapterInitConfig, TransformersAdapter
from cad.calibrator import CADCalibrator
from cad.decoder import CADConfig, ContextAwareDecoder
from cad.discovery import NegativePromptBuilder


def _read_text(value: str | None, path: str | None, label: str) -> str:
    if value and path:
        raise ValueError(f"Use either --{label} or --{label}-file, not both.")
    if path:
        return Path(path).read_text().strip()
    if value:
        return value.strip()
    raise ValueError(f"Provide --{label} or --{label}-file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fincad",
        description="Apply FinCAD with a pre-computed discovery JSON.",
    )
    parser.add_argument("--model-name", required=True, help="Hugging Face model id or local model path")
    parser.add_argument("--discovery-file", required=True, help="Path to results/discovery/<model>.json")
    parser.add_argument("--context", help="Evidence available at the historical decision time")
    parser.add_argument("--context-file", help="Read historical context from a UTF-8 text file")
    parser.add_argument("--task", help="Task instruction and required output format")
    parser.add_argument("--task-file", help="Read the task instruction from a UTF-8 text file")
    parser.add_argument("--entity", default="", help="Entity name or ticker, e.g. NVDA")
    parser.add_argument("--date", default="", help="Historical decision date in YYYY-MM-DD format")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed CAD strength. Omit to use entity/date-adaptive calibration.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--attn-implementation", default=None)
    template_group = parser.add_mutually_exclusive_group()
    template_group.add_argument(
        "--use-chat-template",
        dest="use_chat_template",
        action="store_true",
        help="Use the model chat template (default)",
    )
    template_group.add_argument(
        "--no-chat-template",
        dest="use_chat_template",
        action="store_false",
        help="Encode prompts as plain completions",
    )
    parser.set_defaults(use_chat_template=True)
    parser.add_argument("--show-prompts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        context = _read_text(args.context, args.context_file, "context")
        task = _read_text(args.task, args.task_file, "task")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.alpha is None and (not args.entity or not args.date):
        raise SystemExit("Adaptive alpha requires both --entity and --date; otherwise pass --alpha.")
    if args.alpha is not None and args.alpha < 0:
        raise SystemExit("--alpha must be non-negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    init_kwargs = {
        "model_name": args.model_name,
        "use_chat_template": args.use_chat_template,
        "cache_dir": args.cache_dir,
        "attn_implementation": args.attn_implementation,
    }
    if args.device:
        init_kwargs["device"] = args.device
    adapter = TransformersAdapter(AdapterInitConfig(**init_kwargs))
    adapter.model.eval()

    builder = NegativePromptBuilder.from_file(args.discovery_file)
    decoder = ContextAwareDecoder(
        adapter.model,
        adapter.tokenizer,
        device=adapter.device,
        use_chat_template=args.use_chat_template,
    )

    alpha = args.alpha
    if alpha is None:
        calibrator = CADCalibrator(
            adapter.model,
            adapter.tokenizer,
            device=adapter.device,
            use_chat_template=args.use_chat_template,
            optimized_instruction=builder.instruction.instruction,
            logit_gap_profile=builder.instruction.logit_gap_profile,
        )
        calibrator.calibrate_entity(args.entity)
        alpha = calibrator.calibrate_alpha(args.entity, date=args.date).alpha

    context_prompt = f"{context}\n\n{task}".strip()
    prior_prompt = builder.build(entity=args.entity, date=args.date, task_prompt=task)
    config = CADConfig(
        alpha=float(alpha),
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    if args.show_prompts:
        print("=== Context prompt ===")
        print(context_prompt)
        print("\n=== FinCAD prior prompt ===")
        print(prior_prompt)
        print()

    output = decoder.generate(context_prompt, prior_prompt, config)
    print(f"FinCAD alpha: {alpha:.4f}")
    print(output)


if __name__ == "__main__":
    main()
