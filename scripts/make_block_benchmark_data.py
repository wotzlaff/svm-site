"""Summarize copied block-working-set records for the Benchmarks page.

Run ``python scripts/make_block_benchmark_data.py`` from the site checkout. Raw CSVs and
build provenance in data/benchmarks/block* come unchanged from smo-solver/benchmarks/results.
Only valid post-fix HPO records are accepted; timed-out solver runs remain censored.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks"
SOLVER_FILES = [
    "block_working_sets.csv",
    "block_working_sets_moderate.csv",
    "block_working_sets_moderate_size4.csv",
]


def rows(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_solver():
    groups = defaultdict(list)
    for filename in SOLVER_FILES:
        for row in rows(filename):
            groups[row["case"], int(row["working_set_size"])].append(row)
    cases = {}
    for (case, size), records in groups.items():
        converged = [r for r in records if r["status"] == "converged"]
        for row in converged:
            assert float(row["recomputed_violation"]) <= 1.01 * float(row["tol"])
            assert math.isfinite(float(row["recomputed_gap"]))
        cases.setdefault(case, {})[str(size)] = {
            "n": int(records[0]["n"]),
            "runs": len(records),
            "converged": len(converged),
            "seconds": min(float(r["seconds"]) for r in converged)
            if converged
            else None,
            "steps": min(int(r["steps"]) for r in converged) if converged else None,
            "gap_range": [
                min(float(r["recomputed_gap"]) for r in records),
                max(float(r["recomputed_gap"]) for r in records),
            ],
            "violation_range": [
                min(float(r["recomputed_violation"]) for r in records),
                max(float(r["recomputed_violation"]) for r in records),
            ],
        }
    return cases


def summarize_hpo():
    groups = defaultdict(list)
    for row in rows("block_hpo.csv"):
        assert row["cv_verified"] == "True" and row["optimizer_success"] == "True"
        assert float(row["cv_error_difference"]) <= 1e-7
        assert float(row["cv_max_norm_excess"]) <= 1e-7
        groups[row["case"], int(row["working_set_size"])].append(row)
    cases = {}
    for (case, size), records in groups.items():
        cases.setdefault(case, {})[str(size)] = {
            "n": int(records[0]["n"]),
            "runs": len(records),
            "seconds": min(float(r["seconds"]) for r in records),
            "cv_error": float(records[0]["cv_error"]),
            "evaluations": int(records[0]["evaluations"]),
            "max_cv_difference": max(float(r["cv_error_difference"]) for r in records),
            "max_norm_excess": max(float(r["cv_max_norm_excess"]) for r in records),
        }
    return cases


def main():
    summary = {
        "source_files": SOLVER_FILES + ["block_hpo.csv"],
        "sizes": [2, 4, 8, 16],
        "solver": summarize_solver(),
        "hpo": summarize_hpo(),
    }
    out = ROOT / "public/data/block-benchmarks.json"
    out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
