"""Profile a model's logit gap distribution for alpha normalization.

Probes the model across representative (entity, date) pairs using the
same completion-based probe as ``CADCalibrator``, collects the raw logit
gaps, and writes summary statistics back into the discovery JSON so that
the calibrator can use model-specific normalization at inference time.

Usage::

    python -m cad.discovery.profiler \
        --model-name /path/to/model \
        --price-csv dataset/backtest-data/price/price_data.csv \
        --optimized-instruction results/discovery/phi-4.json \
        --alpha-target 3.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Profile logit gap distribution for a model")
    p.add_argument("--model-name", required=True, help="HF model name or path")
    p.add_argument("--price-csv", required=True, help="Price data CSV")
    p.add_argument("--optimized-instruction", required=True,
                   help="Path to discovery JSON (will be updated in-place)")
    p.add_argument("--alpha-target", type=float, default=3.0,
                   help="Target alpha for P95 gap (default: 3.0)")
    p.add_argument("--in-sample-start", default="2005-01-01")
    p.add_argument("--in-sample-end", default="2015-01-01")
    p.add_argument("--oos-start", default="2025-01-01",
                   help="Out-of-sample start date")
    p.add_argument("--oos-end", default="2026-01-01",
                   help="Out-of-sample end date")
    p.add_argument("--cache-dir", "--model-cache-dir", default=None)
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--use-chat-template", action="store_true")
    return p.parse_args(argv)


def build_probe_pairs(price_csv: str, in_sample_range: tuple, oos_range: tuple):
    """Build (entity, date) pairs for profiling.

    Returns two lists: in_sample_pairs and oos_pairs.
    """
    from .calibration_data import build_calibration_dataset
    from .config import CalibrationDatasetConfig

    # In-sample: reuse the same calibration dataset builder
    cfg = CalibrationDatasetConfig(
        price_csv=price_csv,
        date_range=in_sample_range,
        forward_days=63,
        max_examples=400,
        sample_freq="QE",
        min_abs_return=0.0,  # keep all, we just need the (entity, date) pairs
    )
    in_sample_examples = build_calibration_dataset(cfg)
    in_sample_pairs = [(ex.ticker, ex.date) for ex in in_sample_examples]
    logger.info("In-sample pairs: %d", len(in_sample_pairs))

    # Out-of-sample: use the same tickers but with dates beyond training cutoff
    tickers = sorted(set(t for t, _ in in_sample_pairs))

    # Generate quarterly dates in OOS range
    oos_start, oos_end = pd.Timestamp(oos_range[0]), pd.Timestamp(oos_range[1])
    oos_dates = pd.date_range(oos_start, oos_end, freq="QE")
    if len(oos_dates) == 0:
        oos_dates = pd.date_range(oos_start, oos_end, freq="ME")

    oos_pairs = [
        (ticker, d.strftime("%Y-%m-%d"))
        for ticker in tickers
        for d in oos_dates
    ]
    logger.info("Out-of-sample pairs: %d (%d tickers × %d dates)",
                len(oos_pairs), len(tickers), len(oos_dates))

    return in_sample_pairs, oos_pairs


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load model
    from adapters import AdapterInitConfig, TransformersAdapter
    from cad.calibrator import CADCalibrator
    from cad.discovery import NegativePromptBuilder

    logger.info("Loading model %s ...", args.model_name)
    adapter = TransformersAdapter(
        AdapterInitConfig(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            attn_implementation=args.attn_implementation,
        )
    )

    # Load T* instruction
    neg_prompt_builder = NegativePromptBuilder.from_file(args.optimized_instruction)
    optimized_instruction = neg_prompt_builder.instruction.instruction

    # Build calibrator (without profile — we're computing the profile)
    calibrator = CADCalibrator(
        adapter.model,
        adapter.tokenizer,
        device=adapter.device,
        use_chat_template=args.use_chat_template,
        optimized_instruction=optimized_instruction,
    )

    # Build probe pairs
    in_sample_pairs, oos_pairs = build_probe_pairs(
        args.price_csv,
        in_sample_range=(args.in_sample_start, args.in_sample_end),
        oos_range=(args.oos_start, args.oos_end),
    )

    # Run probes and collect gaps + entropy
    all_gaps = []
    in_sample_gaps = []
    oos_gaps = []
    all_entropies = []
    in_sample_entropies = []
    oos_entropies = []

    def probe_pairs(pairs, label):
        gaps = []
        entropies = []
        for i, (ticker, date) in enumerate(pairs):
            result = calibrator.calibrate_alpha(ticker, date=date)
            gaps.append(abs(result.p_yes - result.p_no))  # use p_yes/p_no gap as raw stat
            entropies.append(result.entropy)
            if (i + 1) % 50 == 0:
                logger.info("[%s] Probed %d/%d", label, i + 1, len(pairs))
        return gaps, entropies

    logger.info("Probing in-sample pairs ...")
    in_sample_gaps, in_sample_entropies = probe_pairs(in_sample_pairs, "in-sample")
    all_gaps.extend(in_sample_gaps)
    all_entropies.extend(in_sample_entropies)

    logger.info("Probing out-of-sample pairs ...")
    oos_gaps, oos_entropies = probe_pairs(oos_pairs, "oos")
    all_gaps.extend(oos_gaps)
    all_entropies.extend(oos_entropies)

    # Compute statistics
    a = np.array(all_gaps)
    a_in = np.array(in_sample_gaps) if in_sample_gaps else np.array([0.0])
    e_all = np.array(all_entropies)
    e_in = np.array(in_sample_entropies) if in_sample_entropies else np.array([1.0])
    e_oos = np.array(oos_entropies) if oos_entropies else np.array([1.0])

    # Auto-computed anomaly threshold: OOS_mean - IS_std
    # The calibrator uses this formula at runtime; we store the ingredients.
    anomaly_threshold = float(e_oos.mean() - e_in.std())

    profile = {
        # Legacy gap stats (kept for backward compat)
        "gap_p95": float(np.percentile(a_in, 95)),
        "gap_mean": float(a.mean()),
        "gap_std": float(a.std()),
        # Entropy stats (calibrator auto-computes threshold from these)
        "entropy_in_sample_mean": float(e_in.mean()),
        "entropy_in_sample_std": float(e_in.std()),
        "entropy_in_sample_p25": float(np.percentile(e_in, 25)),
        "entropy_in_sample_p50": float(np.percentile(e_in, 50)),
        "entropy_oos_mean": float(e_oos.mean()),
        "entropy_oos_std": float(e_oos.std()),
        "entropy_oos_p05": float(np.percentile(e_oos, 5)),
        "entropy_oos_p50": float(np.percentile(e_oos, 50)),
        # Alpha config
        "alpha_max": 12.0,
        "alpha_min": 0.0,
        "alpha_target": args.alpha_target,
        # Metadata
        "n_probes": len(all_gaps),
        "n_in_sample": len(in_sample_gaps),
        "n_out_of_sample": len(oos_gaps),
        "profiled_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Entropy Profile: {args.model_name}")
    print("=" * 60)
    print(f"  Total probes:     {profile['n_probes']}")
    print(f"  In-sample:        {profile['n_in_sample']}")
    print(f"  Out-of-sample:    {profile['n_out_of_sample']}")
    print(f"\n  Entropy distribution:")
    print(f"    In-sample:  mean={e_in.mean():.3f}  std={e_in.std():.3f}  "
          f"p25={np.percentile(e_in, 25):.3f}  p50={np.percentile(e_in, 50):.3f}")
    print(f"    OOS:        mean={e_oos.mean():.3f}  std={e_oos.std():.3f}  "
          f"p05={np.percentile(e_oos, 5):.3f}  p50={np.percentile(e_oos, 50):.3f}")
    print(f"\n  Auto-computed anomaly threshold (OOS_mean - IS_std): {anomaly_threshold:.3f}")
    print(f"  → Probes above this get α=0 (within OOS confidence range)")
    print(f"  → Probes below this get α scaled up (anomalously confident)")
    amax = 12.0
    print(f"\n  Alpha mapping (α_max={amax:.0f}, cap=3.0, threshold={anomaly_threshold:.3f}):")
    for h_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if h_val < anomaly_threshold:
            confidence = 1.0 - h_val / anomaly_threshold
            a_val = min(amax * confidence, 3.0)
            print(f"    H={h_val:.1f} → α={a_val:.2f}")
        else:
            print(f"    H={h_val:.1f} → α=0.00 (above threshold)")
    print("=" * 60 + "\n")

    # Update discovery JSON in-place
    json_path = Path(args.optimized_instruction)
    with open(json_path) as f:
        discovery_data = json.load(f)

    discovery_data["logit_gap_profile"] = profile

    with open(json_path, "w") as f:
        json.dump(discovery_data, f, indent=2)

    logger.info("Profile written to %s", json_path)


if __name__ == "__main__":
    main()
