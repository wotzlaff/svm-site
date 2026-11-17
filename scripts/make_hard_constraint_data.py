"""Generate the hard-constraint explorer from real optimized-svm fits.

Run with the updated optimized-svm source on PYTHONPATH:
    python scripts/make_hard_constraint_data.py

The JSON records predictions, feasibility, support vectors and constraint violations.
No optimizer is run in the browser. Positive lambda invariance is checked before export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from optimized_svm import SmoothSVC, SmoothSVR


def rounded(a):
    return np.asarray(a).round(7).tolist()


def main():
    grid = np.linspace(-1.8, 1.8, 181)[:, None]
    epsilons = [0.05, 0.15, 0.3, 0.65, 0.9]
    scenarios = []
    for kind in ("classification", "regression"):
        for conflict in (False, True):
            if kind == "classification":
                x = np.array([-1.4, -1.0, -0.6, 0.6, 1.0, 1.4])[:, None]
                y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
            else:
                x = np.linspace(-1.4, 1.4, 9)[:, None]
                y = np.sin(1.8 * x[:, 0])
            if conflict:
                x = np.concatenate([x, [[0.0], [0.0]]])
                y = np.concatenate(
                    [y, [-1.0, 1.0] if kind == "classification" else [-0.6, 0.6]]
                )
            frames = []
            for epsilon in epsilons if kind == "regression" else [None]:
                cls = SmoothSVC if kind == "classification" else SmoothSVR
                args = (
                    {}
                    if epsilon is None
                    else {"epsilon": epsilon, "normalize_y": False}
                )
                soft = cls(scaling=1.0, smoothing=0.0, lam=4.0, **args).fit(x, y)
                predict = "decision_function" if kind == "classification" else "predict"
                expected_feasible = not conflict or (
                    epsilon is not None and epsilon >= 0.6
                )
                frame = {
                    "epsilon": epsilon,
                    "soft": rounded(getattr(soft, predict)(grid)),
                }
                loss = (
                    "hard_margin"
                    if kind == "classification"
                    else "hard_epsilon_insensitive"
                )
                try:
                    hard = cls(loss=loss, scaling=1.0, lam=1.0, **args).fit(x, y)
                except ValueError as exc:
                    if expected_feasible or "infeasible" not in str(exc):
                        raise
                    frame.update(
                        hard=None, support=[], violation=None, lambda_delta=None
                    )
                else:
                    assert expected_feasible, (
                        "An impossible constraint system was accepted"
                    )
                    values = getattr(hard, predict)(grid)
                    at_train = getattr(hard, predict)(x)
                    violation = max(
                        0.0,
                        float(
                            np.max(
                                1.0 - y * at_train
                                if epsilon is None
                                else np.abs(at_train - y) - epsilon
                            )
                        ),
                    )
                    assert violation < 1e-5, violation
                    differences = []
                    for lam in (0.01, 100.0):
                        other = cls(loss=loss, scaling=1.0, lam=lam, **args).fit(x, y)
                        differences.append(
                            np.max(np.abs(getattr(other, predict)(grid) - values))
                        )
                    delta = float(max(differences))
                    assert delta < 1e-5, delta
                    frame.update(
                        hard=rounded(values),
                        support=hard.support_.tolist(),
                        violation=violation,
                        lambda_delta=delta,
                    )
                frames.append(frame)
            scenarios.append(
                {
                    "kind": kind,
                    "conflict": conflict,
                    "x": rounded(x[:, 0]),
                    "y": rounded(y),
                    "frames": frames,
                }
            )
    out = Path(__file__).resolve().parents[1] / "public/data/hard-constraints.json"
    out.write_text(
        json.dumps(
            {"grid": rounded(grid[:, 0]), "epsilons": epsilons, "scenarios": scenarios},
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"Wrote {out} ({out.stat().st_size:,} bytes); feasibility and lambda invariance checked."
    )


if __name__ == "__main__":
    main()
