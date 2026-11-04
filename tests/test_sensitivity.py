import numpy as np
import pytest
from scipy.stats import qmc

from kinetic_hype import sensitivity as sa


def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def test_sobol_estimators_recover_ishigami_indices():
    """Analytic indices of the Ishigami function (a=7, b=0.1)."""
    d, n = 3, 2 ** 14
    u = qmc.Sobol(d=2 * d, scramble=True, seed=1).random(n)
    A, B = (u[:, :d] * 2 - 1) * np.pi, (u[:, d:] * 2 - 1) * np.pi
    fAB = []
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB.append(ishigami(ABi))
    res = sa.sobol_indices(ishigami(A), ishigami(B), np.array(fAB), n_boot=50)
    assert res["S1"] == pytest.approx([0.3139, 0.4424, 0.0], abs=0.02)
    assert res["ST"] == pytest.approx([0.5576, 0.4424, 0.2437], abs=0.02)


def test_baseline_reproduces_paper_run():
    out = sa.evaluate(sa.BASELINE, T=20.0, n_points=2001)
    assert out["hype_cycle"] == 1.0
    assert out["drop"] == pytest.approx(0.879 - 0.729, abs=2e-3)
    assert out["rho_T"] == pytest.approx(0.736, abs=5e-3)


def test_design_shapes_and_scaling():
    design = sa.saltelli_design(8, seed=0)
    assert design["A"].shape == (8, len(sa.FACTORS))
    pts = sa.to_points(design["A"])
    for x in pts:
        for f in sa.FACTORS:
            assert f.low <= x[f.name] <= f.high
    # AB_i differs from A only in column i
    i = 3
    diff = design[f"AB{i}"] != design["A"]
    assert diff[:, i].all() and not np.delete(diff, i, axis=1).any()
