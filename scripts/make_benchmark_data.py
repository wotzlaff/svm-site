"""Condenses the raw benchmark results in `data/benchmarks/` into `public/data/bench.json`.

The raw files come from the benchmark scripts of the other repositories:

* `libsvm.log`            smo-solver:    python -u benchmarks/bench_vs_libsvm.py --sizes 1000 2000 5000
* `hpo_vs_search.json`    optimized-svm: python benchmarks/hpo_vs_search.py --seeds 3
* `timing_{old,new}_{1t,dt}.json`
                          optimized-svm: python benchmarks/hpo_timing.py (once per release and BLAS setting)
* `switch.csv`            optimized-svm: python benchmarks/smo_newton_switch.py --out switch.csv
* `shrinking_{scaling,params,tune}.csv`, `stopping.csv`
                          smo-solver:    python benchmarks/shrinking_study.py {scaling,params,tune},
                                         python benchmarks/stopping_study.py run
* `working_set_{hits,timing,validation}.csv`, `convergence_{lambda,size}.csv`
                          smo-solver:    python benchmarks/working_set_study.py run,
                                         python benchmarks/convergence_study.py {lambda,size} > convergence_<study>.csv
                                         (copied unchanged; the timings come from a loaded machine)
* `working_set_traj_base.csv`, `convergence_{relations_base,factor,smooth}.csv`
                          smo-solver:    working_set_study.py run (traj), convergence_study.py {relations,factor,smooth};
                                         the originals are 9 MB, 1.1 MB, 2.7 MB and 0.4 MB, so the files here are thinned copies.
* `primal_first_order_traces.csv`, `primal_first_order_stats.csv`, `primal_first_order_{smo,timing,ref,run_gd,run_pgd,run_pgd_w1}.csv`,
  `newton_vs_smo.csv`
                          optimized-svm: python benchmarks/primal_first_order.py run --aux --methods ""  (reference, SMO, timing),
                                         python benchmarks/primal_first_order.py run --methods gd   (also pgd, pgd_w1; 20 min each),
                                         python benchmarks/newton_vs_smo.py run > benchmarks/results/newton_vs_smo.csv
                                         (Sec. 10.3; the small files are copied unchanged; the timings come from a loaded machine).

The thinned copies are made from the full smo-solver and optimized-svm results by

    python scripts/make_benchmark_data.py --thin ../smo-solver/benchmarks/results [../optimized-svm/benchmarks/results]

(see `thin()` below):

* `working_set_traj_base.csv`: only the base setting (lambda = 1e-3, gamma = 30) and the columns `step`, `r_min`, `rba_min`,
  `gap_min` (running minima of r, r_diamond and the gap); about 200 steps per rule, log-spaced (first recorded step at or
  after each point of a geometric grid from 1 to the last step, plus the last one).
* `convergence_relations_base.csv`: only lambda = 1e-3, gamma = 30. All rows with a gap value (the O(n^2) measures gap, D, V,
  about 280 of them) are kept, of the every-25th-step rows (r, tau, dpsi, rt) every k-th with k chosen to leave at most 200.
* `primal_first_order_traces.csv`: the three traces (`primal_first_order_{gd,pgd,pgd_w1}.csv.gz`, 3000-3700 rows each) with the columns
  `step`, `dist`, `dist_alpha`, `dist_b`, `psi_gap`, `F`, `grad`; per method the union of a log-spaced grid of 200 steps and a log-spaced
  grid of 100 values of F (the first record at or below each value), about 250 rows. `primal_first_order_stats.csv` holds numbers
  computed from the full traces (final values, first step below a distance, ranges of the error-bound ratios and slopes), because
  they would change with the thinning.
* `convergence_factor.csv`, `convergence_smooth.csv`: the same per lambda; for the factor study additionally only r in
  [1e-9, 1e-3], the range of the c(lambda) fits. `convergence_study.py summary` fits on all rows; the constants agree to a
  few percent, see `fit_check` in this file (run with --thin).

Usage: python scripts/make_benchmark_data.py                 # bench.json from data/benchmarks/
       python scripts/make_benchmark_data.py --thin DIR      # refresh the thinned copies from the full smo-solver results in DIR
       python scripts/make_benchmark_data.py --thin DIR DIR2 # ... and the Sec. 10.3 files from the optimized-svm results in DIR2
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
import sys
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


def shrinking() -> dict:
    out: dict = {"scaling": {}, "params": {}}
    rows = list(csv.DictReader((RAW / "shrinking_scaling.csv").open()))
    for task in ("classification", "regression"):
        sel = [r for r in rows if r["task"] == task]
        ns = sorted({int(r["n"]) for r in sel})
        tols = ["0.01", "0.001", "1e-05"]
        speedup = []
        for tol in tols:
            line = []
            for n in ns:
                t = {r["variant"]: float(r["seconds"]) for r in sel if int(r["n"]) == n and float(r["tol"]) == float(tol)}
                line.append(round(t["off"] / t["default"], 3))
            speedup.append(line)
        out["scaling"][task] = {"n": ns, "tols": ["1e-2", "1e-3", "1e-5"], "speedup": speedup}
    rows = list(csv.DictReader((RAW / "shrinking_params.csv").open()))
    for key, task, lam, tol in (("classification", "classification", 0.01, 1e-5), ("regression", "regression", 0.01, 1e-3)):
        sel = [r for r in rows if r["task"] == task and float(r["lam"]) == lam and float(r["tol"]) == tol and r["variant"] == "threshold"]
        etas = sorted({float(r["exponent"]) for r in sel})
        cs = sorted({float(r["factor"]) for r in sel})
        t = {(float(r["exponent"]), float(r["factor"])): float(r["seconds"]) for r in sel}
        best = min(t.values())
        out["params"][key] = {
            "rows": [f"η = {e:g}" for e in etas],
            "cols": [f"{c:g}" for c in cs],
            "values": [[round(t[(e, c)] / best, 3) for c in cs] for e in etas],
        }
    rows = list(csv.DictReader((RAW / "shrinking_tune.csv").open()))
    groups: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        groups[(r["task"], r["gamma"], r["lam"], r["tol"])][r["variant"]] = float(r["seconds"])
    variants = []
    for name, key in (("earlier default (η = 0.5)", "default"), ("η = 1", "c1.0"), ("η = 1 + minimal fraction (default now)", "min0.1")):
        sp = [g["off"] / g[key] for g in groups.values()]
        variants.append({"name": name, "gmean": round(math.exp(sum(map(math.log, sp)) / len(sp)), 3), "worst": round(min(sp), 3)})
    out["defaults"] = {"variants": variants, "settings": len(groups)}
    return out


def stopping() -> dict:
    rows = list(csv.DictReader((RAW / "stopping.csv").open()))
    pts = [
        {"task": r["task"], "violation": float(r["violation"]), "gap": float(r["gap"]), "f_err": float(r["f_err"])}
        for r in rows
        if float(r["violation"]) > 0 and float(r["f_err"]) > 0
    ]
    return {"points": pts, "n": 1500}


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

# ---------------------------------------------------------------------------------------------
# Sec. 10.2: working sets and convergence of SMO

WS_RULES = ["mvp", "bamvp", "mud", "md", "tmd"]
WS_SETTINGS = {"base": "λ = 10⁻³, γ = 30", "lam1e-2": "λ = 10⁻²", "lam1e-4": "λ = 10⁻⁴", "gamma10": "γ = 10"}
FIT_RANGE = (1e-9, 1e-3)  # range of r used for the fits, as in the thesis (Sec. 10.2.5)
# measure -> (column, power of r, thesis fit (a, s) of c(lambda) = a lambda^s)
MEASURES = {
    "gap": ("gap", 1, (5.770, -0.04)),
    "tau": ("tau", 1, (1.648, 0.958)),
    "dpsi": ("dpsi", 2, (0.427, 0.989)),
    "rt": ("rt", 1, (0.707, 0.014)),
}


def _num(x: str) -> float:
    return float(x) if x != "" else math.nan


def _src_open(src: Path, name: str):
    """Opens `src / name` for reading, or its gzipped version `name.gz` (the large traces)."""
    f = src / name
    if not f.exists() and f.with_name(name + ".gz").exists():
        return gzip.open(f.with_name(name + ".gz"), "rt", newline="")
    return f.open(newline="")


def thin(src: Path) -> None:
    """Writes the thinned copies in `data/benchmarks/` from the full smo-solver results in `src`."""
    for name in ("working_set_hits.csv", "working_set_timing.csv", "working_set_validation.csv", "convergence_lambda.csv", "convergence_size.csv"):
        (RAW / name).write_bytes((src / name).read_bytes())

    def write(name: str, fields: list[str], rows: list[dict]) -> None:
        with (RAW / name).open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(fields)
            for r in rows:
                w.writerow([r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in fields])
        print(f"wrote {RAW / name} ({(RAW / name).stat().st_size} bytes)")

    # running minima on a logarithmic grid of steps
    rows = [r for r in csv.DictReader(_src_open(src, "working_set_traj.csv")) if r["setting"] == "base"]
    out = []
    for rule in WS_RULES:
        rs = [r for r in rows if r["rule"] == rule]
        last = int(rs[-1]["step"])
        grid = [round(math.exp(math.log(max(last, 2)) * i / 199)) for i in range(200)]
        keep, j = [], 0
        for r in rs:
            if int(r["step"]) >= grid[j]:
                keep.append(r)
                while j < len(grid) and grid[j] <= int(r["step"]):
                    j += 1
                if j == len(grid):
                    break
        if keep[-1] is not rs[-1]:
            keep.append(rs[-1])
        out += [{"rule": rule, "step": r["step"], "r_min": float(r["r_min"]), "rba_min": float(r["rba_min"]), "gap_min": float(r["gap_min"])} for r in keep]
    write("working_set_traj_base.csv", ["rule", "step", "r_min", "rba_min", "gap_min"], out)

    fields = ["lam", "step", "r", "tau", "dpsi", "rt", "gap", "D", "V"]
    for study, name, only_base, in_range in (
        ("relations", "convergence_relations_base.csv", True, False),
        ("factor", "convergence_factor.csv", False, True),
        ("smooth", "convergence_smooth.csv", False, False),
    ):
        sel = []
        for r in csv.DictReader(_src_open(src, f"convergence_{study}.csv")):
            if only_base and (float(r["lam"]) != 1e-3 or float(r["gamma"]) != 30.0):
                continue
            if in_range and not FIT_RANGE[0] <= float(r["r"]) <= FIT_RANGE[1]:
                continue
            sel.append(r)
        out = []
        for lam, rs in _by(sel, "lam").items():
            with_gap = [r for r in rs if r["gap"] != ""]  # the O(n^2) measures: kept completely
            rest = [r for r in rs if r["gap"] == ""]  # every 25th step: every k-th row
            k = max(1, math.ceil(len(rest) / 200))
            keep = sorted(with_gap + rest[::k], key=lambda r: int(r["step"]))
            out += [{f: (r[f] if f in ("step",) else (float(r[f]) if r[f] != "" else "")) for f in fields} for r in keep]
        write(name, fields, out)

PRIMAL_METHODS = ["gd", "pgd", "pgd_w1"]
PRIMAL_COPIED = ["primal_first_order_smo.csv", "primal_first_order_timing.csv", "primal_first_order_ref.csv", "newton_vs_smo.csv"] + [
    f"primal_first_order_run_{m}.csv" for m in PRIMAL_METHODS
]
TRACE_FIELDS = ["step", "dist", "dist_alpha", "dist_b", "psi_gap", "F", "grad"]


def _slope(xs: list[float], ys: list[float]) -> float:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    return _linfit([p[0] for p in pts], [p[1] for p in pts])[1] if len(pts) > 5 else math.nan


def _trace_stats(rows: list[dict]) -> dict[str, float | str]:
    """Numbers of one primal trace (full resolution), see `cmd_summary` in optimized-svm/benchmarks/primal_first_order.py."""
    f = lambda k: [float(r[k]) for r in rows]  # noqa: E731
    step, dist, F, gap, grad, da, db, tau, time = (f(k) for k in ("step", "dist", "F", "psi_gap", "grad", "dist_alpha", "dist_b", "tau", "time"))
    out: dict[str, float | str] = {"rows": len(rows), "steps": step[-1], "time": time[-1], "dist_final": dist[-1], "dist_alpha_final": da[-1], "dist_b_final": db[-1],
                                   "dist_start": dist[0], "dist_b_start": db[0], "gap_final": gap[-1], "F_final": F[-1]}  # fmt: skip
    taus = sorted(t for t in tau if not math.isnan(t))
    out["tau_median"] = taus[len(taus) // 2]
    for exp in range(1, 7):
        thr = 10.0**-exp
        i = next((i for i, d in enumerate(dist) if d <= thr), None)
        if i is not None:
            out[f"first_dist_{exp}_step"], out[f"first_dist_{exp}_time"] = step[i], time[i]
    idx = [i for i in range(len(rows)) if F[i] < 1e-2 and gap[i] > 1e-11 and dist[i] > 0]  # asymptotic region of the error bound
    out["bound_rows"] = len(idx)
    if len(idx) >= 6:
        for name, num, den in (("F_dist", F, dist), ("grad_dist", grad, dist), ("db_da", db, da)):
            rr = sorted(num[i] / den[i] for i in idx)
            out[f"{name}_min"], out[f"{name}_max"], out[f"{name}_median"] = rr[0], rr[-1], rr[len(rr) // 2]
        sl = [i for i in idx if 1e-6 <= F[i] <= 1e-2]
        for name, v in (("gap", gap), ("dist", dist), ("grad", grad), ("dist_alpha", da), ("dist_b", db)):
            out[f"slope_{name}"] = _slope([F[i] for i in sl], [v[i] for i in sl])
    return out


def thin_primal(src: Path) -> None:
    """Writes the Sec. 10.3 files in `data/benchmarks/` from the optimized-svm results in `src`."""
    for name in PRIMAL_COPIED:
        (RAW / name).write_bytes((src / name).read_bytes())
    traces, stats = [], []
    for m in PRIMAL_METHODS:
        with gzip.open(src / f"primal_first_order_{m}.csv.gz", "rt", newline="") as fh:
            rows = list(csv.DictReader(fh))
        stats += [(m, k, v if isinstance(v, str) else f"{v:.6g}") for k, v in _trace_stats(rows).items()]
        last = int(float(rows[-1]["step"]))
        keep: set[int] = set()
        grid = [round(math.exp(math.log(max(last, 2)) * i / 199)) for i in range(200)]  # log-spaced steps
        j = 0
        for i, r in enumerate(rows):
            if int(float(r["step"])) >= grid[j]:
                keep.add(i)
                while j < len(grid) and grid[j] <= int(float(r["step"])):
                    j += 1
                if j == len(grid):
                    break
        f0, f1 = float(rows[0]["F"]), float(rows[-1]["F"])  # log-spaced values of F: the first record at or below each
        j, levels = 0, [f0 * (f1 / f0) ** (i / 99) for i in range(100)]
        for i, r in enumerate(rows):
            if float(r["F"]) <= levels[j]:
                keep.add(i)
                while j < len(levels) and levels[j] >= float(r["F"]):
                    j += 1
                if j == len(levels):
                    break
        keep.add(len(rows) - 1)
        traces += [{"method": m, **{k: rows[i][k] for k in TRACE_FIELDS}} for i in sorted(keep)]
    with (RAW / "primal_first_order_traces.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", *TRACE_FIELDS])
        for r in traces:
            w.writerow([r["method"], int(float(r["step"])), *(f"{float(r[k]):.6g}" for k in TRACE_FIELDS[1:])])
    with (RAW / "primal_first_order_stats.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["method", "key", "value"])
        w.writerows(stats)
    for name in ("primal_first_order_traces.csv", "primal_first_order_stats.csv"):
        print(f"wrote {RAW / name} ({(RAW / name).stat().st_size} bytes)")


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = a + s x; returns (a, s, R^2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    a = my - s * mx
    syy = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - a - s * x) ** 2 for x, y in zip(xs, ys, strict=True))
    return a, s, 1 - ss_res / syy if syy > 0 else 1.0


def _measure_fit(rows: list[dict], col: str, power: int) -> dict | None:
    """c in v ~ c r^power (geometric mean of v / r^power over the fit range) and the local exponent."""
    pts = [(float(r["r"]), float(r[col])) for r in rows if r[col] != "" and FIT_RANGE[0] <= float(r["r"]) <= FIT_RANGE[1] and float(r[col]) > 0]
    if len(pts) < 5:
        return None
    c = math.exp(sum(math.log(v / r**power) for r, v in pts) / len(pts))
    _, slope, _ = _linfit([math.log(r) for r, _ in pts], [math.log(v) for _, v in pts])
    return {"c": c, "slope": slope}


def _by(rows: list[dict], key: str) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        out[float(r[key])].append(r)
    return dict(sorted(out.items()))


def _sig(x: float, d: int = 4) -> float:
    return float(f"{x:.{d}g}")


def working_sets() -> dict:
    hits = list(csv.DictReader((RAW / "working_set_hits.csv").open()))
    # first step at which a measure is <= threshold; None if it was not reached
    first = {(r["setting"], r["rule"], r["measure"], float(r["threshold"])): int(r["step"]) for r in hits if r["step"] != ""}
    steps = {
        s: {
            rule: {m: {t: first.get((s, rule, m, float(t))) for t in ("1e-3", "1e-5", "1e-8")} for m in ("r", "rba_lam", "gap")}
            for rule in WS_RULES
        }
        for s in WS_SETTINGS
    }
    timing = list(csv.DictReader((RAW / "working_set_timing.csv").open()))
    us = {
        f"{r['rule']}_{r['impl']}": _sig(float(r["us_per_step"]), 3)
        for r in timing
        if r["setting"] == "base" and ((r["kind"] == "to_tol" and r["threshold"] == "1e-08") or r["kind"] == "per_step")
    }
    traj: dict = {}
    for r in csv.DictReader((RAW / "working_set_traj_base.csv").open()):
        t = traj.setdefault(r["rule"], {"step": [], "r": [], "rba": [], "gap": []})
        t["step"].append(int(r["step"]))
        t["r"].append(_sig(float(r["r_min"])))
        t["rba"].append(_sig(float(r["rba_min"])))
        t["gap"].append(_sig(float(r["gap_min"])))
    valid = list(csv.DictReader((RAW / "working_set_validation.csv").open()))
    return {
        "settings": WS_SETTINGS,
        "steps": steps,
        "us_per_step": us,
        "traj": traj,
        "validation": {
            "max_alpha_diff": max(float(r["max_alpha_diff"]) for r in valid),
            "mvp_steps": min(int(r["steps"]) for r in valid if r["rule"] == "mvp"),
            "first_mismatch": [int(r["first_pair_mismatch"]) for r in valid if r["first_pair_mismatch"] != ""],
        },
    }


def _scatter_series(rows: list[dict], points: bool = True, gap_power: int = 1) -> dict:
    """Per measure the pairs (r, value) of the rows that have the measure (gap, D, V and the others are recorded on different steps)."""
    out: dict = {}
    for key in ("gap", "tau", "dpsi", "D") if points else ():
        pts = [(float(r["r"]), float(r[key])) for r in rows if r[key] != "" and float(r[key]) > 0 and float(r["r"]) > 0]
        out[key] = {"r": [_sig(a, 3) for a, _ in pts], "v": [_sig(b, 3) for _, b in pts]}
    out["fit"] = {k: _measure_fit(rows, col, gap_power if k == "gap" else p) for k, (col, p, _) in MEASURES.items()}
    out["fit"].update({"D": _measure_fit(rows, "D", 1), "V": _measure_fit(rows, "V", 2)})
    return out


def convergence() -> dict:
    out: dict = {}
    rel = list(csv.DictReader((RAW / "convergence_relations_base.csv").open()))
    out["hinge"] = _scatter_series(rel)
    sm = list(csv.DictReader((RAW / "convergence_smooth.csv").open()))
    out["smooth"] = {f"{lam:g}": _scatter_series(rs, points=lam == 1e-3, gap_power=2) for lam, rs in _by(sm, "lam").items()}
    # c(lambda): per lambda the geometric-mean factor, then a log-log line through them
    fac = list(csv.DictReader((RAW / "convergence_factor.csv").open()))
    byl = _by(fac, "lam")
    lams = list(byl)
    factor: dict = {"lam": lams, "measures": {}}
    for key, (col, p, thesis) in MEASURES.items():
        cs = [_measure_fit(rs, col, p)["c"] for rs in byl.values()]  # type: ignore[index]
        b, s, r2 = _linfit([math.log(x) for x in lams], [math.log(c) for c in cs])
        factor["measures"][key] = {"c": [_sig(c) for c in cs], "a": _sig(math.exp(b)), "s": _sig(s), "r2": _sig(r2), "thesis": {"a": thesis[0], "s": thesis[1]}}
    # ratio r / gap, the thesis uses 0.173
    ratios = [1 / _measure_fit(rs, "gap", 1)["c"] for rs in byl.values()]  # type: ignore[index]
    factor["r_over_gap"] = _sig(sum(ratios) / len(ratios))
    out["factor"] = factor
    # support vectors and free variables
    lam_rows = list(csv.DictReader((RAW / "convergence_lambda.csv").open()))
    out["lambda"] = {
        f"{d:g}": {
            "lam": [float(r["lam"]) for r in rs],
            "n_sv": [int(r["n_sv"]) for r in rs],
            "n_free": [int(r["n_free"]) for r in rs],
            "n_upper": [int(r["n_upper"]) for r in rs],
        }
        for d, rs in _by(lam_rows, "delta").items()
    }
    # size: steps (mean), time (min over repetitions), free variables (mean) over n for each lambda
    size_rows = list(csv.DictReader((RAW / "convergence_size.csv").open()))
    size: dict = {"tol": float(size_rows[0]["tol"]), "lam": {}}
    for lam, rs in _by(size_rows, "lam").items():
        ns = sorted({int(r["n"]) for r in rs})
        pick = lambda col, agg: [agg(float(r[col]) for r in rs if int(r["n"]) == n) for n in ns]  # noqa: E731
        steps, secs, free = pick("steps", statistics.fmean), pick("seconds", min), pick("n_free", statistics.fmean)
        exps = [_linfit([math.log(n) for n in ns], [math.log(v) for v in vs])[1] for vs in (steps, secs, free)]
        size["lam"][f"{lam:g}"] = {
            "n": ns, "steps": [round(v) for v in steps], "seconds": [_sig(v) for v in secs], "free": [round(v) for v in free],
            "exp_steps": _sig(exps[0], 3), "exp_time": _sig(exps[1], 3), "exp_free": _sig(exps[2], 3),
        }  # fmt: skip
    out["size"] = size
    return out


# ---------------------------------------------------------------------------------------------
# Sec. 10.3: first-order primal methods, error bound, Newton versus SMO


def primal() -> dict:
    traces: dict = {}
    for r in csv.DictReader((RAW / "primal_first_order_traces.csv").open()):
        t = traces.setdefault(r["method"], {k: [] for k in ("step", "dist", "dist_alpha", "dist_b", "psi_gap", "F", "grad")})
        t["step"].append(int(r["step"]))
        for k in ("dist", "dist_alpha", "dist_b", "psi_gap", "F", "grad"):
            t[k].append(_sig(float(r[k]), 3))
    stats: dict = defaultdict(dict)
    for r in csv.DictReader((RAW / "primal_first_order_stats.csv").open()):
        stats[r["method"]][r["key"]] = r["value"] if r["key"] == "stop_reason" else float(r["value"])
    for r in (r for m in PRIMAL_METHODS for r in csv.DictReader((RAW / f"primal_first_order_run_{m}.csv").open())):
        stats[r["method"]]["stop_reason"] = r["stop_reason"]
    smo = [
        {"tol": float(r["tol"]), "steps": int(r["steps"]), "time": float(r["time"]), "dist": float(r["dist"])}
        for r in csv.DictReader((RAW / "primal_first_order_smo.csv").open())
    ]
    # SMO: first tolerance of the sweep with distance <= 1e-3 (the sweep is the tolerance sequence of the script)
    smo_hit = next(r for r in smo if r["dist"] <= 1e-3)
    timing = {r["method"]: float(r["s_per_step"]) for r in csv.DictReader((RAW / "primal_first_order_timing.csv").open())}
    return {"traces": traces, "stats": dict(stats), "smo": smo, "smo_1e-3": smo_hit, "s_per_step": timing}


def newton() -> dict:
    rows = list(csv.DictReader((RAW / "newton_vs_smo.csv").open()))
    ns = sorted({int(r["n"]) for r in rows})
    out: dict = {
        "n": ns,
        "reps": int(rows[0]["reps"]),
        "kernel": [_sig(float(next(r["t_kernel"] for r in rows if int(r["n"]) == n))) for n in ns],
        "failed": sum(int(r["nw_failed"]) + int(r["two_failed"]) + int(r["smo_timeout"]) for r in rows),
        "lam": {},
    }
    for lam, rs in _by(rows, "lam").items():
        rs.sort(key=lambda r: int(r["n"]))
        col = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        two = [a + b for a, b in zip(col("two_t_smo"), col("two_t_newton"), strict=True)]
        series = {"newton": col("nw_t"), "smo": col("smo_t"), "two": two}
        big = [i for i, r in enumerate(rs) if int(r["n"]) >= 1000]  # exponents of power laws t ~ n^e from n >= 1000
        exps = {k: _linfit([math.log(int(rs[i]["n"])) for i in big], [math.log(v[i]) for i in big])[1] for k, v in series.items()}
        out["lam"][f"{lam:g}"] = {
            "newton_steps": [int(r["nw_steps"]) for r in rs],
            "newton_t": [_sig(v, 3) for v in series["newton"]],
            "smo_t": [_sig(v, 3) for v in series["smo"]],
            "two_t": [_sig(v, 3) for v in two],
            "two_smo_t": [_sig(v, 3) for v in col("two_t_smo")],
            "two_newton_t": [_sig(v, 3) for v in col("two_t_newton")],
            "exp_newton": _sig(exps["newton"], 3), "exp_smo": _sig(exps["smo"], 3), "exp_two": _sig(exps["two"], 3),
        }  # fmt: skip
    return out


def fit_check(full: Path) -> None:
    """Compares the c(lambda) fits from the thinned copy with those from the full factor file (every 25th step included)."""
    rows = list(csv.DictReader(_src_open(full, "convergence_factor.csv")))
    thin_fit = convergence()["factor"]["measures"]
    for key, (col, p, _) in MEASURES.items():
        cs = [_measure_fit(rs, col, p)["c"] for rs in _by(rows, "lam").values()]  # type: ignore[index]
        lams = list(_by(rows, "lam"))
        b, s, _ = _linfit([math.log(x) for x in lams], [math.log(c) for c in cs])
        print(f"{key}: full c = {math.exp(b):.4g} lam^{s:.3f}; thinned c = {thin_fit[key]['a']:.4g} lam^{thin_fit[key]['s']:.3f}")


if __name__ == "__main__":
    if "--thin" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--thin") + 1])
        thin(src)
        fit_check(src)
        rest = sys.argv[sys.argv.index("--thin") + 2 :]
        if rest and not rest[0].startswith("--"):
            thin_primal(Path(rest[0]))
        raise SystemExit
    data = {"libsvm": libsvm(), "search": search(), "versions": versions(), "switch": switch(), "shrinking": shrinking(), "stopping": stopping(), "workingsets": working_sets(), "convergence": convergence(), "primal": primal(), "newton": newton()}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
