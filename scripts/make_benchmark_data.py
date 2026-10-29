"""Condenses the raw benchmark results in `data/benchmarks/` into `public/data/bench.json`.

The raw files come from the benchmark scripts of the other repositories:

* `libsvm.log`            smo-solver:    python -u benchmarks/bench_vs_libsvm.py --sizes 1000 2000 5000
* `hpo_vs_search.json`    optimized-svm: python benchmarks/hpo_vs_search.py --seeds 3
* `timing_{old,new}_{1t,dt}.json`
                          optimized-svm: python benchmarks/hpo_timing.py (once per release and BLAS setting)
* `switch.csv`            optimized-svm: python benchmarks/smo_newton_switch.py --out switch.csv

Usage: python scripts/make_benchmark_data.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "benchmarks"
OUT = ROOT / "public" / "data" / "bench.json"


def libsvm() -> list[dict]:
    rows = []
    for line in (RAW / "libsvm.log").read_text().splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[0] in ("classification", "regression"):
            task, n, ours, ref, steps, ref_steps, _ = parts
            rows.append(
                {
                    "task": task,
                    "n": int(n),
                    "ours_s": float(ours),
                    "libsvm_s": float(ref),
                    "ours_steps": int(steps),
                    "libsvm_steps": int(ref_steps),
                }
            )
    return rows


def search() -> dict:
    res = json.loads((RAW / "hpo_vs_search.json").read_text())
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in res:
        by[(r["dataset"], r["method"])].append(r)
    datasets: dict[str, dict] = {}
    for (dataset, method), rs in by.items():
        d = datasets.setdefault(dataset, {"task": rs[0]["task"], "methods": {}})
        # classification: report the error rate; regression: the MSE (smaller is better)
        scores = [(1 - r["score"]) if r["task"] == "cls" else r["score"] for r in rs]
        d["methods"][method] = {
            "seconds": statistics.mean(r["seconds"] for r in rs),
            "evaluations": statistics.mean(r["evaluations"] for r in rs),
            "error": statistics.mean(scores),
            "error_sd": statistics.pstdev(scores),
            "runs": len(rs),
        }
    return datasets


def versions() -> dict:
    out = {}
    for blas in ("1t", "dt"):
        old = json.loads((RAW / f"timing_old_{blas}.json").read_text())
        new = json.loads((RAW / f"timing_new_{blas}.json").read_text())
        out[blas] = {
            "old_version": old["optimized_svm"],
            "new_version": new["optimized_svm"],
            "old_solver": old["smo_solver"],
            "new_solver": new["smo_solver"],
            "cases": [
                {"case": o["case"], "old_s": o["seconds"], "new_s": n["seconds"],
                 "old_err": o["cv_error"], "new_err": n["cv_error"]}
                for o, n in zip(old["cases"], new["cases"], strict=True)
            ],
        }  # fmt: skip
    return out


def superscript(k: int) -> str:
    return str(k).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def switch() -> dict:
    rows = list(csv.DictReader((RAW / "switch.csv").open()))
    tols = sorted({float(r["tol"]) for r in rows})
    out = {"cols": ["0.001", "0.01", "0.03", "0.1", "0.3", "1", "3", "no SMO"], "tasks": {}}
    for task in ("svc", "svr"):
        for scenario in ("cold", "warm_step"):
            sel = [
                r for r in rows
                if r["task"] == task and r["scenario"] == scenario and r["n"] == "1000"
                and float(r["s_factor"]) == 1.0 and float(r["smoothing"]) == 0.5
                and r["degree"] == "2" and r["max_asum"] == ""
            ]  # fmt: skip
            lams = sorted({float(r["lam"]) for r in sel}, reverse=True)
            values = []
            for lam in lams:
                row = {float(r["tol"]): float(r["t_total"]) for r in sel if float(r["lam"]) == lam}
                best = min(row.values())
                values.append([round(row[t] / best, 3) for t in tols])
            out["tasks"].setdefault(task, {})[scenario] = {
                "rows": [f"λ = 10{superscript(round(math.log10(lam)))}" for lam in lams],
                "values": values,
            }
    return out


if __name__ == "__main__":
    data = {"libsvm": libsvm(), "search": search(), "versions": versions(), "switch": switch()}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
