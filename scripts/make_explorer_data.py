"""Precomputes the data of the interactive explainers on /explore/ with optimized-svm.

Run from the repository root with an environment that has `optimized-svm`:

    python scripts/make_explorer_data.py

Writes `public/data/hpo-path.json`, `gradient-slice.json` and `ard-path.json`.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from optimized_svm import HingeLoss, OptimizedSVC, OptimizedSVR, SmoothSVC, cv_error_and_gradient
from optimized_svm.validation import SmoothHingeValidation

OUT = Path(__file__).resolve().parent.parent / "public" / "data"
N = 300
GRID = 40  # resolution of the stored decision functions


def checkerboard(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) > 0, 1, -1)
    return x, np.where(rng.random(n) < 0.05, -y, y)


def grid_row(lg, lams, x, labels, folds):
    """CV error and exact gradient for one value of log10 gamma (warm start: large to small lambda).

    Returns three lists: error, d error / d log10(lambda) and d error / d log10(gamma). The
    hyperparameters of the optimizer are theta = (ln lambda, ln s) with gamma = s^2.
    """
    loss, validation = HingeLoss(smoothing=0.5), SmoothHingeValidation()
    warm = [None] * len(folds)
    err, g_lam, g_gam = [], [], []
    for ll in lams[::-1]:
        theta = np.array([ll * np.log(10), 0.5 * lg * np.log(10)])
        res = cv_error_and_gradient(
            theta, x, labels, np.ones(len(labels)), folds, loss, validation, warm_start=warm
        )
        err.append(res.error)
        g_lam.append(res.gradient[0] * np.log(10))
        g_gam.append(res.gradient[1] * 0.5 * np.log(10))
    return err[::-1], g_lam[::-1], g_gam[::-1]


def r(a, digits: int = 4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def hpo_path() -> None:
    x, y = checkerboard(N)
    labels = y.astype(float)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    lams = np.linspace(-4.5, 0, 25)
    gammas = np.linspace(0, 3, 25)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(partial(grid_row, lams=lams, x=x, labels=labels, folds=folds), gammas))
    err, g_lam, g_gam = (np.array([row[k] for row in rows]) for k in range(3))
    model = OptimizedSVC(lam_init=1e-1, scaling_init=1.5, cv=folds).fit(x, y)
    gx, gy = np.meshgrid(np.linspace(0, 1, GRID), np.linspace(0, 1, GRID))
    pts = np.c_[gx.ravel(), gy.ravel()]
    iterates = []
    for h in model.history_:
        lam, s = float(h["lam"]), float(h["scaling"][0])
        est = SmoothSVC(lam=lam, scaling=s).fit(x, y)
        dec = np.clip(est.decision_function(pts), -3, 3)
        iterates.append(
            {
                "log10_lam": float(np.log10(lam)),
                "log10_gamma": float(np.log10(s**2)),
                "error": float(h["error"]),
                "decision": r(dec, 2),
            }
        )
    best = int(np.argmin(err))
    data = {
        "grid": GRID,
        "log10_lam": r(lams, 3),
        "log10_gamma": r(gammas, 3),
        "cv_error": r(err, 5),
        "grad_log10_lam": r(g_lam, 5),
        "grad_log10_gamma": r(g_gam, 5),
        "grid_best": {
            "log10_lam": float(lams[best % len(lams)]),
            "log10_gamma": float(gammas[best // len(lams)]),
            "error": float(err.flat[best]),
        },
        "points": [[round(float(a), 4), round(float(b), 4), int(c)] for (a, b), c in zip(x, y, strict=True)],
        "iterates": iterates,
    }
    (OUT / "hpo-path.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"hpo-path: {len(iterates)} iterates, grid {err.shape}, best grid error {err.min():.4f}")


def gradient_slice() -> None:
    x, y = checkerboard(N)
    labels = y.astype(float)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    loss, validation = HingeLoss(smoothing=0.5), SmoothHingeValidation()
    s = 5.0  # kernel scaling (gamma = s^2 = 25), near the optimum of the checkerboard problem
    log10_lams = np.linspace(-4.5, 0, 91)
    errs, grads = [], []
    warm: list[np.ndarray | None] = [None] * len(folds)
    for ll in log10_lams[::-1]:
        theta = np.array([ll * np.log(10), np.log(s)])
        res = cv_error_and_gradient(
            theta, x, labels, np.ones(len(labels)), folds, loss, validation, warm_start=warm
        )
        errs.append(res.error)
        grads.append(res.gradient[0] * np.log(10))  # d error / d log10(lambda) = ln(10) d error / d ln(lambda)
    data = {
        "scaling": s,
        "log10_lam": r(log10_lams, 4),
        "cv_error": r(errs[::-1], 6),
        "d_log10_lam": r(grads[::-1], 6),
    }
    (OUT / "gradient-slice.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"gradient-slice: {len(errs)} points, min error {min(errs):.4f}")


def ard_path() -> None:
    """ARD on synthetic data: y = 10 sin(pi x0 x1) + 20 (x2 - 1/2)^2 + noise, five noise features."""
    rng = np.random.default_rng(2)
    n, p = 400, 8
    x = rng.random((n, p))
    y = 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2 + 0.5 * rng.normal(size=n)
    x = StandardScaler().fit_transform(x)
    model = OptimizedSVR(blocks="features", two_stage=True, cv=5, max_iter=60).fit(x, y)
    rows = [h for h in model.history_ if h["stage"] == "features"]
    data = {
        "n_features": p,
        "relevant": [0, 1, 2],
        "iterates": [
            {
                "error": float(h["error"]),
                "log10_scaling": r(np.log10(np.maximum(h["scaling"], 1e-6)), 3),
                "log10_lam": float(np.log10(h["lam"])),
            }
            for h in rows
        ],
    }
    (OUT / "ard-path.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"ard-path: {len(rows)} iterates; final scalings {np.round(model.scaling_opt_, 3)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hpo_path()
    gradient_slice()
    ard_path()
