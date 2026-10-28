"""Generates the social preview image ``public/og.png`` (1200 x 630).

Run with an environment that has `optimized-svm` and `matplotlib`:

    python scripts/make_og.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from optimized_svm import SmoothSVC  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "public" / "og.png"
BG, FG, MUTED, BLUE, ORANGE = "#f7f8fb", "#141a26", "#5b6577", "#1d4ed8", "#c2410c"

rng = np.random.default_rng(3)
x = rng.random((400, 2))
y = np.where(np.sin(4 * np.pi * x[:, 0]) * np.sin(4 * np.pi * x[:, 1]) > 0, 1, -1)
y = np.where(rng.random(400) < 0.04, -y, y)
clf = SmoothSVC(lam=1e-3, scaling=6.0).fit(x, y)

g = np.linspace(0, 1, 300)
gx, gy = np.meshgrid(g, g)
z = clf.decision_function(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
ax = fig.add_axes([0.64, 0.1, 0.30, 0.8])
ax.set_aspect("equal")
ax.contourf(gx, gy, z, levels=[-99, 0, 99], colors=[ORANGE, BLUE], alpha=0.16)
ax.contour(gx, gy, z, levels=[0], colors=[FG], linewidths=2)
ax.scatter(*x[y > 0].T, s=14, color=BLUE)
ax.scatter(*x[y < 0].T, s=14, color=ORANGE)
ax.set_xticks([])
ax.set_yticks([])
for s in ax.spines.values():
    s.set_color("#dde2ea")

fig.text(0.05, 0.62, "Support vector machines", fontsize=31, fontweight="bold", color=FG,
         family="serif")
fig.text(0.05, 0.52, "duality & exact hyperparameter gradients", fontsize=18, color=BLUE,
         family="serif")
fig.text(0.05, 0.34, "Interactive demo · Rust SMO solver · Python package", fontsize=14,
         color=MUTED)
fig.text(0.05, 0.1, "Nico Strasdat · TU Dresden", fontsize=15, color=MUTED)
fig.savefig(OUT, facecolor=BG)
