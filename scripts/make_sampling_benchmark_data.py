"""Generate the website summary from the public sampling benchmark CSV.

Run from the site checkout with `python scripts/make_sampling_benchmark_data.py`.
Ratios and score differences are calculated within each outer split before
taking medians. Classification scores are kept as fractions until display.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

DATA = Path(__file__).resolve().parents[1] / "public/data"
DATASETS = {
    "breast_cancer": (
        "Breast Cancer Wisconsin (Diagnostic)",
        "real",
        "accuracy",
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
    ),
    "diabetes": (
        "Diabetes",
        "real",
        "rmse",
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html",
    ),
    "moons": ("Noisy two moons", "synthetic", "accuracy", None),
    "friedman1": ("Friedman #1", "synthetic", "rmse", None),
}
METHODS = {
    "cv5": "Full 5-fold CV",
    "holdout": "Fixed holdout",
    "short_cv": "Short 5-fold CV",
    "subset_scaled": "Subset + short CV (scaled λ)",
    "subset_direct": "Subset + short CV (direct λ)",
}
SEEDS = tuple(range(5))


def summarize(rows):
    by_key = {}
    for row in rows:
        key = (row["dataset"], row["protocol"], int(row["seed"]))
        if key in by_key:
            raise ValueError(f"Duplicate record: {key}")
        seconds, score = float(row["fit_seconds"]), float(row["test_score"])
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(score):
            raise ValueError(f"Invalid timing or test score: {key}")
        by_key[key] = row
    expected = {(d, m, s) for d in DATASETS for m in METHODS for s in SEEDS}
    if by_key.keys() != expected:
        raise ValueError(f"Unexpected record set: {by_key.keys() ^ expected}")

    datasets = []
    for dataset, (label, kind, metric, source) in DATASETS.items():
        baseline = [by_key[dataset, "cv5", seed] for seed in SEEDS]
        samples = int(baseline[0]["samples"])
        methods = []
        for method, method_label in METHODS.items():
            records = [by_key[dataset, method, seed] for seed in SEEDS]
            for base, record in zip(baseline, records, strict=True):
                if (
                    record["metric"] != metric
                    or int(record["samples"]) != samples
                    or any(
                        record[key] != base[key]
                        for key in ("outer_train", "outer_test")
                    )
                ):
                    raise ValueError(
                        f"Incompatible record for {dataset}/{method}/{record['seed']}"
                    )
            times = [float(row["fit_seconds"]) for row in records]
            scores = [float(row["test_score"]) for row in records]
            speedups = [
                float(base["fit_seconds"]) / seconds
                for base, seconds in zip(baseline, times, strict=True)
            ]
            deltas = [
                score - float(base["test_score"])
                for base, score in zip(baseline, scores, strict=True)
            ]
            methods.append(
                {
                    "id": method,
                    "label": method_label,
                    "seconds": median(times),
                    "score": median(scores),
                    "speedup": median(speedups),
                    "speedup_min": min(speedups),
                    "speedup_max": max(speedups),
                    "delta": median(deltas),
                    "delta_min": min(deltas),
                    "delta_max": max(deltas),
                    "converged": sum(row["converged"] == "True" for row in records),
                    "capped": sum(
                        row["converged"] == "False"
                        and "ITERATIONS REACHED LIMIT" in row["optimizer_message"]
                        for row in records
                    ),
                }
            )
        datasets.append(
            {
                "id": dataset,
                "label": label,
                "kind": kind,
                "source": source,
                "samples": samples,
                "metric": metric,
                "seeds": len(SEEDS),
                "methods": methods,
            }
        )
    return {"datasets": datasets}


def main():
    raw = DATA / "sampling/hpo_sampling_web.csv"
    with raw.open(encoding="utf-8", newline="") as stream:
        summary = summarize(csv.DictReader(stream))
    out = DATA / "sampling-benchmarks.json"
    out.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
