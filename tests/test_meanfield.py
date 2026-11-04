import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc

from kinetic_hype import Params, Signals, fixed_signals, meanfield as mf

P0 = Params()
S0 = Signals()
STATES = [(0.0, 0.4, 0.5), (0.2, 0.6, 2.0), (0.5, 0.7, 6.0), (0.9, 0.79, 15.0)]


@pytest.mark.parametrize("rho,r,t", STATES)
def test_hazard_is_tail_of_stationary_density(rho, r, t):
    P = S0.price(t)
    tail = quad(lambda v: mf.stationary_density(v, rho, r, t, P0, S0), P, np.inf)[0]
    assert mf.adoption_hazard(rho, r, t, P0, S0) == pytest.approx(tail, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("rho,r,t", STATES)
def test_density_mean_equals_m1(rho, r, t):
    """Self-consistency: the stationary density's mean is the fast-scale fixed point m1."""
    mean = quad(lambda v: v * mf.stationary_density(v, rho, r, t, P0, S0), 0, np.inf, limit=200)[0]
    assert mean == pytest.approx(mf.mean_nonusers(rho, r, t, P0, S0), rel=1e-6)


def legacy_analytic_hazard(rho, r, t, a1, a2, b, gamma, s):
    """compute_adoption_rate_gamma from the original Analytic_Hype.ipynb (hazard only)."""
    mu, w, P = s.media(t), s.message(t), s.price(t)
    m_val = P * (a2 * rho * r + mu * b * w) / (a2 * rho + mu * b + 1e-12)
    alpha1, alpha2 = a1 / b, a2 / b
    shape = 1.0 + 2.0 * (1.0 + (1.0 - rho) * alpha1 + rho * alpha2) / gamma
    scale = (2.0 / gamma) * ((1 - rho) * alpha1 * (m_val / P) + alpha2 * rho * r + w)
    return gammainc(shape, scale)


@pytest.mark.parametrize("rho,r", [(0.0, 0.4), (0.3, 0.6), (0.8, 0.79)])
def test_equals_unit_media_and_original_code_when_media_is_one(rho, r):
    s = fixed_signals(media=1.0)
    ours = mf.adoption_hazard(rho, r, 3.0, P0, s)
    assert ours == pytest.approx(mf.adoption_hazard_unit_media(rho, r, 3.0, P0, s), rel=1e-9)
    assert ours == pytest.approx(legacy_analytic_hazard(rho, r, 3.0, P0.a1, P0.a2, P0.b, P0.gam, s),
                                 rel=1e-6)


def test_differs_from_unit_media_when_media_recedes():
    s = fixed_signals(media=0.05)
    assert abs(mf.adoption_hazard(0.5, 0.7, 3.0, P0, s)
               - mf.adoption_hazard_unit_media(0.5, 0.7, 3.0, P0, s)) > 1e-3


def test_m1_limits():
    s = fixed_signals(media=1.0, message=1.2)
    assert mf.mean_nonusers(0.0, 0.5, 1.0, P0, s) == pytest.approx(1.2)      # media only
    s0 = fixed_signals(media=0.0, message=1.2)
    assert mf.mean_nonusers(0.4, 0.5, 1.0, P0, s0) == pytest.approx(0.5)     # users only


def test_deterministic_limit():
    """As gam -> 0 the hazard tends to the indicator m1 > P."""
    s = fixed_signals(media=1.0, message=1.2)                                # m1 = 1.2 > P at rho = 0
    small = Params(gam=1e-3)
    assert mf.adoption_hazard(0.0, 0.4, 1.0, small, s) > 0.999
    s_low = fixed_signals(media=1.0, message=0.9)                             # m1 = 0.9 < P
    assert mf.adoption_hazard(0.0, 0.4, 1.0, small, s_low) < 1e-6
    det = Params(gam=0.0)
    assert mf.adoption_hazard(0.0, 0.4, 1.0, det, s) == 1.0
    assert mf.adoption_hazard(0.0, 0.4, 1.0, det, s_low) == 0.0


def test_simulation_is_well_behaved():
    d = mf.simulate(P0, S0, T=20)
    assert (np.diff(d["rho"]) >= -1e-12).all()
    assert d["rho"].between(0, 1).all()
    assert d["r"].between(0, P0.r_inf).all()


def test_deterministic_simulation_runs():
    d = mf.simulate(Params(gam=0.0), S0, T=10)
    assert (np.diff(d["rho"]) >= -1e-12).all()
