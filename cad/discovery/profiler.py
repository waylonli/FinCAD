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

    # Run probes and collect entropy + delta_temporal, grouped by entity
    from collections import defaultdict
    in_sample_entropies = []
    oos_entropies = []
    in_sample_deltas = []
    oos_deltas = []
    # Per-entity tracking for date-variance analysis
    entity_is_entropies = defaultdict(list)  # ticker → [H values across dates]

    def probe_pairs(pairs, label, track_entity=False):
        entropies = []
        deltas = []
        for i, (ticker, date) in enumerate(pairs):
            result = calibrator.calibrate_alpha(ticker, date=date)
            entropies.append(result.entropy)
            deltas.append(result.delta_temporal)
            if track_entity:
                entity_is_entropies[ticker].append(result.entropy)
            if (i + 1) % 50 == 0:
                logger.info("[%s] Probed %d/%d", label, i + 1, len(pairs))
        return entropies, deltas

    # Pre-calibrate all entities so delta_temporal is meaningful
    unique_tickers = sorted(set(t for t, _ in in_sample_pairs))
    logger.info("Calibrating %d entities ...", len(unique_tickers))
    for ticker in unique_tickers:
        calibrator.calibrate_entity(ticker)

    logger.info("Probing in-sample pairs ...")
    in_sample_entropies, in_sample_deltas = probe_pairs(
        in_sample_pairs, "in-sample", track_entity=True)

    logger.info("Probing out-of-sample pairs ...")
    oos_entropies, oos_deltas = probe_pairs(oos_pairs, "oos")

    # Compute per-entity date-variance: std of entropy across dates for each entity
    # High std = confidence varies with date = date-specific memorisation
    # Low std = confidence stable across dates = brand prior
    entity_stds = []
    for ticker, hs in entity_is_entropies.items():
        if len(hs) >= 2:
            entity_stds.append(np.std(hs))
    entity_stds = np.array(entity_stds) if entity_stds else np.array([0.0])

    # Compute statistics
    e_in = np.array(in_sample_entropies) if in_sample_entropies else np.array([1.0])
    e_oos = np.array(oos_entropies) if oos_entropies else np.array([1.0])
    d_in = np.array(in_sample_deltas) if in_sample_deltas else np.array([0.0])
    d_oos = np.array(oos_deltas) if oos_deltas else np.array([0.0])

    profile = {
        # Entropy stats (kept for reference)
        "entropy_in_sample_mean": float(e_in.mean()),
        "entropy_in_sample_std": float(e_in.std()),
        "entropy_in_sample_p25": float(np.percentile(e_in, 25)),
        "entropy_in_sample_p50": float(np.percentile(e_in, 50)),
        "entropy_oos_mean": float(e_oos.mean()),
        "entropy_oos_std": float(e_oos.std()),
        "entropy_oos_p05": float(np.percentile(e_oos, 5)),
        "entropy_oos_p50": float(np.percentile(e_oos, 50)),
        # Date-ablation delta stats
        "delta_temporal_is_mean": float(d_in.mean()),
        "delta_temporal_is_std": float(d_in.std()),
        "delta_temporal_oos_mean": float(d_oos.mean()),
        "delta_temporal_oos_std": float(d_oos.std()),
        # Per-entity date-variance stats (new — for α scaling)
        # Entities with low date-variance have brand priors, not memorisation
        "entity_date_variance_mean": float(entity_stds.mean()),
        "entity_date_variance_std": float(entity_stds.std()),
        "entity_date_variance_p50": float(np.percentile(entity_stds, 50)),
        "entity_date_variance_p75": float(np.percentile(entity_stds, 75)),
        "entity_date_variance_p90": float(np.percentile(entity_stds, 90)),
        "n_entities_profiled": len(entity_is_entropies),
        # Alpha config
        "alpha_max": 12.0,
        "alpha_min": 0.0,
        "alpha_target": args.alpha_target,
        # Metadata
        "n_probes": len(in_sample_entropies) + len(oos_entropies),
        "n_in_sample": len(in_sample_entropies),
        "n_out_of_sample": len(oos_entropies),
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
    print(f"\n  Date-ablation delta (Δ = H_undated - H_dated):")
    print(f"    In-sample:  mean={d_in.mean():+.3f}  std={d_in.std():.3f}")
    print(f"    OOS:        mean={d_oos.mean():+.3f}  std={d_oos.std():.3f}")
    print(f"\n  Per-entity date-variance (std of entropy across dates):")
    print(f"    Across {len(entity_is_entropies)} entities:")
    print(f"    mean={entity_stds.mean():.4f}  std={entity_stds.std():.4f}  "
          f"p50={np.percentile(entity_stds, 50):.4f}  p75={np.percentile(entity_stds, 75):.4f}  "
          f"p90={np.percentile(entity_stds, 90):.4f}")
    print(f"    → High variance = date-specific memorisation (penalise)")
    print(f"    → Low variance = stable brand prior (do not penalise)")
    # Show top-5 most date-varying entities
    entity_std_map = {t: np.std(hs) for t, hs in entity_is_entropies.items() if len(hs) >= 2}
    sorted_entities = sorted(entity_std_map.items(), key=lambda x: -x[1])
    print(f"    Top-5 date-varying: {', '.join(f'{t}({s:.3f})' for t,s in sorted_entities[:5])}")
    print(f"    Bottom-5 (stable):  {', '.join(f'{t}({s:.3f})' for t,s in sorted_entities[-5:])}")
    print(f"\n  Calibrator config:")
    print(f"    Δ_oos_baseline (OOS mean + OOS std): {d_oos.mean() + d_oos.std():+.3f}")
    print(f"    Δ_range (IS std):          {d_in.std():.3f}")
    amax = 12.0
    print(f"\n  Alpha mapping (α_max={amax:.0f}, cap=3.0):")
    for d_val in [-0.1, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        excess = d_val - d_oos.mean()
        if excess <= 0:
            a_val = 0.0
        else:
            a_val = min(amax * excess / max(d_in.std(), 0.05), 3.0)
        print(f"    Δ={d_val:+.2f} → excess={excess:+.3f} → α={a_val:.2f}")
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
