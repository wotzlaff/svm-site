"""Generates the figures of the website with optimized-svm (light and dark variants).

Run from the repository root with an environment that has `optimized-svm` and `matplotlib`:

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from optimized_svm import HingeLoss, OptimizedSVC, cv_error_and_gradient  # noqa: E402
from optimized_svm.validation import SmoothHingeValidation  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "public" / "figures"
THEMES = {
    "light": {"fg": "#1d2433", "muted": "#5b6577", "grid": "#d7dce5", "a": "#c2410c",
              "b": "#1d4ed8", "path": "#0f172a", "cmap": "Blues_r"},
    "dark": {"fg": "#e6e9ef", "muted": "#9aa4b5", "grid": "#323a4a", "a": "#fb923c",
             "b": "#60a5fa", "path": "#f8fafc", "cmap": "Blues_r"},
}


def checkerboard(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dataset of Sec. 10.1: labels sign(sin(4 pi x1) sin(4 pi x2)) on the unit square."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    f = np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1])
    y = np.where(f > 0, 1, -1)
    flip = rng.random(n) < 0.05
    return x, np.where(flip, -y, y)


def style(ax, t):
    ax.tick_params(colors=t["muted"], labelsize=9)
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x, y = checkerboard(500)
    labels = y.astype(float)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    loss, validation = HingeLoss(smoothing=0.5), SmoothHingeValidation()

    # CV error landscape over (log10 lam, log10 gamma), gamma = s^2
    lams = np.linspace(-5, 0, 21)
    gammas = np.linspace(0, 3, 21)
    err = np.zeros((gammas.size, lams.size))
    warm = [None] * len(folds)
    for i, lg in enumerate(gammas):
        for j, ll in enumerate(lams):
            theta = np.array([ll * np.log(10), 0.5 * lg * np.log(10)])
            err[i, j] = cv_error_and_gradient(theta, x, labels, np.ones(len(y)), folds, loss,
                                              validation, warm_start=warm).error
        print(f"grid row {i + 1}/{gammas.size}", flush=True)

    model = OptimizedSVC(lam_init=1e-1, scaling_init=1.5, cv=folds).fit(x, y)
    path = np.array([[np.log10(h["lam"]), np.log10(h["scaling"][0] ** 2)] for h in model.history_])
    print(f"optimum: lam = {model.lam_:.3g}, gamma = {model.scaling_opt_[0] ** 2:.3g}, "
          f"{len(model.history_)} evaluations, train acc {model.score(x, y):.3f}")

    for name, t in THEMES.items():
        # landscape
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        cs = ax.contourf(lams, gammas, np.log10(err), levels=18, cmap=t["cmap"])
        ax.contour(lams, gammas, np.log10(err), levels=18, colors=t["grid"], linewidths=0.4)
        ax.plot(path[:, 0], path[:, 1], "-o", color=t["path"], ms=3.5, lw=1.4)
        ax.plot(*path[0], "o", color=t["a"], ms=8, label="start")
        ax.plot(*path[np.argmin([h["error"] for h in model.history_])], "*", color=t["a"],
                ms=14, label="optimum")
        ax.set_xlabel(r"$\log_{10}\lambda$")
        ax.set_ylabel(r"$\log_{10}\gamma$")
        cb = fig.colorbar(cs, ax=ax)
        cb.set_label(r"$\log_{10}$ CV error", color=t["fg"])
        cb.ax.tick_params(colors=t["muted"], labelsize=8)
        cb.outline.set_edgecolor(t["grid"])
        leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_color(t["fg"])
        style(ax, t)
        fig.tight_layout()
        fig.savefig(OUT / f"hpo-landscape-{name}.svg", transparent=True)
        plt.close(fig)

        # decision function of the optimized model
        g = np.linspace(0, 1, 200)
        gx, gy = np.meshgrid(g, g)
        dec = model.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.contour(gx, gy, dec, levels=[0], colors=t["fg"], linewidths=1.6)
        ax.scatter(*x[y > 0].T, s=9, marker="o", color=t["b"], alpha=0.7, linewidths=0)
        ax.scatter(*x[y < 0].T, s=12, marker="x", color=t["a"], alpha=0.7, linewidths=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        style(ax, t)
        fig.tight_layout()
        fig.savefig(OUT / f"checkerboard-{name}.svg", transparent=True)
        plt.close(fig)

    summary = {
        "lam": model.lam_,
        "gamma": float(model.scaling_opt_[0] ** 2),
        "evaluations": len(model.history_),
        "grid_points": int(err.size),
        "cv_error": model.cv_error_,
    }
    (OUT / "hpo-summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
