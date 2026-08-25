#!/usr/bin/env python3
"""Recreate the exhaustive seven-of-eleven ranking-alignment analysis."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon

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
                alternative = "greater" if condition == "cad" else "less"
                entry["wilcoxon_alternative"] = alternative
                entry["wilcoxon_vs_baseline_p"] = float(
                    wilcoxon(rho, baseline_rho, alternative=alternative).pvalue
                )
            metric_report[condition] = entry
        report[metric] = metric_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
