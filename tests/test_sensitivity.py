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
    assert out["hype_cycle"] == 1.0 and out["ends_above_peak"] == 0.0
    assert out["drop"] == pytest.approx(0.879 - 0.729, abs=2e-3)
    assert out["rho_T"] == pytest.approx(0.736, abs=5e-3)


def test_cycle_when_message_below_ceiling():
    """w < r_inf: no overshoot of the final level, but a fall after the media peak and a recovery
    above it, which counts as a hype cycle."""
    x = dict(sa.BASELINE, w0=0.81, r_inf=0.86, media_width2=1.83, alpha2=7.79, lam=1.07, gam=0.26,
             k1=0.52, inv_slope=0.12)
    out = sa.evaluate(x)
    assert out["pattern"] == 2.0 and out["hype_cycle"] == 1.0 and out["ends_above_peak"] == 1.0
    assert out["overshoot"] <= 1e-9          # global peak is the final level
    assert out["max_fall"] > 0.03


def test_no_fall_after_media_peak_while_message_below_performance():
    """After the media peak theta = b mu / (a2 rho + b mu) cannot rise, and w(t), r(t) cannot fall, so
    dm1/dt = theta'(w - r) + theta w' + (1 - theta) r' >= 0 whenever w(t) <= r(t)."""
    from kinetic_hype import meanfield
    rng = np.random.default_rng(3)
    checked = 0
    for x in sa.to_points(rng.random((300, len(sa.FACTORS)))):
        p, s = sa.build(x)
        d = meanfield.simulate(p, s, T=30, n_points=601)
        after = (d.t >= sa.MEDIA_PEAK).to_numpy()
        below = (d.message * d.price <= d.r).to_numpy()
        dm = np.diff(d.m1.to_numpy())
        mask = (after & below)[:-1] & (after & below)[1:]
        if mask.any():
            checked += 1
            assert dm[mask].min() > -1e-6
    assert checked > 20


def test_largest_fall_on_synthetic_curves():
    import pandas as pd
    from kinetic_hype.analysis import largest_fall
    t = np.linspace(0, 10, 1001)
    dip = 0.4 * np.exp(-(t - 2) ** 2) + 0.9 * (1 - np.exp(-t / 3))     # early bump, then higher plateau
    f = largest_fall(pd.DataFrame({"t": t, "m1": dip}))
    assert f.exceeded and f.t_peak == pytest.approx(2, abs=0.2) and f.fall > 0.1
    mono = 1 - np.exp(-t)
    assert np.isnan(largest_fall(pd.DataFrame({"t": t, "m1": mono})).fall)
    hump = np.exp(-(t - 3) ** 2) + 0.3                                  # global peak, never exceeded
    g = largest_fall(pd.DataFrame({"t": t, "m1": hump}))
    assert not g.exceeded and g.fall == pytest.approx(1.0, abs=1e-3)


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
