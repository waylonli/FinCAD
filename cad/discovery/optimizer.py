"""Orchestrate DSPy prompt optimization for the memory-activation instruction."""
from __future__ import annotations

import logging
import os
from typing import Optional

from .config import CalibrationDatasetConfig, DiscoveryConfig, OptimizedInstruction
from .calibration_data import build_calibration_dataset, to_dspy_examples
from .metrics import bias_activation_score
from .registry import save_instruction

try:
    import dspy
except ImportError:
    dspy = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def run_optimization(cfg: DiscoveryConfig) -> OptimizedInstruction:
    """Run the full optimization pipeline.

    1. Build calibration data from price CSV.
    2. Split 80/20 into train/val.
    3. Configure ``dspy.LM`` for the target model.
    4. Run MIPROv2 (or COPRO) to optimize the MemoryProbe instruction.
    5. Evaluate on val set and save the result.
    """
    if dspy is None:
        raise ImportError("dspy is required. Install with: pip install dspy")

    from .modules import MemoryProbeModule

    # Step 1: Build calibration dataset
    logger.info("Building calibration dataset from %s ...", cfg.calibration.price_csv)
    examples = build_calibration_dataset(cfg.calibration)
    logger.info("Built %d calibration examples", len(examples))

    if len(examples) < 5:
        raise ValueError(
            f"Only {len(examples)} calibration examples found. "
            "Need at least 5. Check date range and price data coverage."
        )

    dspy_examples = to_dspy_examples(examples)

    # Step 2: Train/val split (80/20)
    split = int(len(dspy_examples) * 0.8)
    trainset = dspy_examples[:split]
    valset = dspy_examples[split:]
    logger.info("Train: %d, Val: %d", len(trainset), len(valset))

    # Step 3: Configure DSPy LM
    # Suppress LiteLLM warning: "`max_retries` is not supported. It will be ignored."
    logging.getLogger("litellm.llms.huggingface.chat.transformation").setLevel(logging.ERROR)

    is_local_path = cfg.model_name.startswith(("/", "~", ".")) or os.path.exists(cfg.model_name)

    if is_local_path and not cfg.server_url:
        raise ValueError(
            f"--model-name looks like a local path ({cfg.model_name}) but no "
            f"--server-url was provided. Local models require an OpenAI-compatible "
            f"server (e.g. vLLM). Start one with:\n"
            f"  python -m vllm.entrypoints.openai.api_server --model {cfg.model_name}\n"
            f"Then pass: --server-url http://localhost:8000/v1"
        )

    if cfg.server_url:
        # Local model served via vLLM / TGI — connect through OpenAI-compatible API
        model_id = f"openai/{cfg.model_name}"
        lm = dspy.LM(model=model_id, api_base=cfg.server_url, api_key="EMPTY")
        logger.info("Configured DSPy LM: %s via %s", model_id, cfg.server_url)
    else:
        # HuggingFace Hub model — use remote Inference API
        model_id = f"huggingface/{cfg.model_name}"
        lm = dspy.LM(model=model_id)
        logger.info("Configured DSPy LM: %s", model_id)
    dspy.configure(lm=lm)

    # Step 4: Run optimizer
    module = MemoryProbeModule()

    if cfg.optimizer == "MIPROv2":
        optimizer = dspy.MIPROv2(
            metric=bias_activation_score,
            auto=None,
            num_candidates=cfg.num_candidates,
            num_threads=1,
        )
        compiled = optimizer.compile(
            module,
            trainset=trainset,
            valset=valset,
            num_trials=cfg.num_trials,
        )
    elif cfg.optimizer == "COPRO":
        optimizer = dspy.COPRO(
            metric=bias_activation_score,
            depth=cfg.num_trials,
        )
        compiled = optimizer.compile(
            module,
            trainset=trainset,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}. Use 'MIPROv2' or 'COPRO'.")

    # Step 5: Extract optimized instruction
    optimized_instruction = compiled.probe.signature.instructions

    # Evaluate on val set
    correct = 0
    for ex in valset:
        try:
            pred = compiled(entity=ex.entity, date=ex.date)
            correct += bias_activation_score(ex, pred)
        except Exception:
            pass
    val_score = correct / len(valset) if valset else 0.0
    logger.info("Val accuracy: %.2f%% (%d/%d)", val_score * 100, int(correct), len(valset))

    result = OptimizedInstruction(
        instruction=optimized_instruction,
        model_name=cfg.model_name,
        score=val_score,
        metadata={
            "optimizer": cfg.optimizer,
            "num_candidates": cfg.num_candidates,
            "num_trials": cfg.num_trials,
            "train_size": len(trainset),
            "val_size": len(valset),
            "forward_days": cfg.calibration.forward_days,
        },
    )

    # Save
    save_instruction(result, output_dir=cfg.output_dir)
    logger.info("Saved optimized instruction to %s", cfg.output_dir)

    return result
