#!/usr/bin/env python3
"""Recreate the descriptive ranking-alignment robustness analysis."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

CONDITIONS = ("baseline", "anonymized", "prompt_injection", "cad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/reproduction/backtests")
    parser.add_argument("--models", default="configs/paper/models.json")
    parser.add_argument("--output", default="results/reproduction/alignment_summary.json")
    return parser.parse_args()


def load_metric(root: Path, condition: str, model: str, period: str, metric: str) -> float:
    path = root / f"{condition}_SPY_{model}_{period}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return float(json.loads(path.read_text())["strategy"][metric])


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    models = list(json.loads(Path(args.models).read_text()))
    if len(models) != 11:
        raise SystemExit(f"Expected 11 paper models, found {len(models)}")

    report: dict[str, dict] = {}
    for metric in ("sharpe", "sortino"):
        oos = {m: load_metric(root, "baseline", m, "out_of_sample", metric) for m in models}
        correlations = {condition: {"spearman": [], "kendall": []} for condition in CONDITIONS}

        for subset in itertools.combinations(models, 7):
            target = [oos[m] for m in subset]
            for condition in CONDITIONS:
                insample = [load_metric(root, condition, m, "in_sample", metric) for m in subset]
                correlations[condition]["spearman"].append(float(spearmanr(insample, target).statistic))
                correlations[condition]["kendall"].append(float(kendalltau(insample, target).statistic))

        metric_report: dict[str, dict] = {}
        baseline_rho = np.asarray(correlations["baseline"]["spearman"])
        baseline_tau = np.asarray(correlations["baseline"]["kendall"])
        for condition in CONDITIONS:
            rho = np.asarray(correlations[condition]["spearman"])
            tau = np.asarray(correlations[condition]["kendall"])
            entry = {
                "mean_spearman": float(rho.mean()),
                "mean_kendall": float(tau.mean()),
                "positive_spearman_fraction": float((rho > 0).mean()),
                "n_subsets": int(len(rho)),
            }
            if condition != "baseline":
                rho_delta = rho - baseline_rho
                tau_delta = tau - baseline_tau
                entry.update(
                    {
                        "mean_spearman_delta_vs_baseline": float(rho_delta.mean()),
                        "median_spearman_delta_vs_baseline": float(np.median(rho_delta)),
                        "spearman_delta_positive_fraction": float((rho_delta > 0).mean()),
                        "spearman_delta_range": [float(rho_delta.min()), float(rho_delta.max())],
                        "mean_kendall_delta_vs_baseline": float(tau_delta.mean()),
                        "median_kendall_delta_vs_baseline": float(np.median(tau_delta)),
                        "kendall_delta_positive_fraction": float((tau_delta > 0).mean()),
                        "kendall_delta_range": [float(tau_delta.min()), float(tau_delta.max())],
                    }
                )
            metric_report[condition] = entry

        lomo = []
        for omitted in models:
            retained = [model for model in models if model != omitted]
            target = [oos[model] for model in retained]
            baseline = [
                load_metric(root, "baseline", model, "in_sample", metric) for model in retained
            ]
            cad = [load_metric(root, "cad", model, "in_sample", metric) for model in retained]
            baseline_rho_lomo = float(spearmanr(baseline, target).statistic)
            cad_rho_lomo = float(spearmanr(cad, target).statistic)
            baseline_tau_lomo = float(kendalltau(baseline, target).statistic)
            cad_tau_lomo = float(kendalltau(cad, target).statistic)
            lomo.append(
                {
                    "omitted_model": omitted,
                    "baseline_spearman": baseline_rho_lomo,
                    "cad_spearman": cad_rho_lomo,
                    "spearman_delta": cad_rho_lomo - baseline_rho_lomo,
                    "baseline_kendall": baseline_tau_lomo,
                    "cad_kendall": cad_tau_lomo,
                    "kendall_delta": cad_tau_lomo - baseline_tau_lomo,
                }
            )

        report[metric] = {
            "seven_model_subset_summary": metric_report,
            "leave_one_model_out": lomo,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
