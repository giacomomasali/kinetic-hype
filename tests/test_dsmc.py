import numpy as np
import pytest
from scipy.stats import invgamma

from kinetic_hype import Params, Signals, dsmc, fixed_signals, meanfield as mf, trajectory_error
from kinetic_hype.dsmc import _update


def test_collision_uses_pre_collision_values_and_conserves_mean():
    """Two non-users, no media, no noise: the pair's mean is conserved and both
    updates use the partner's value before the collision."""
    p = Params(a1=0.1, b=0.001, gam=0.0)
    rng = np.random.default_rng(0)
    x, y = np.array([1.0]), np.array([3.0])
    users = np.array([False])
    nx = _update(x, y, users, users, p, 0.0, 1.2, 1.0, 0.5, 0.0, 0.0, rng)
    ny = _update(y, x, users, users, p, 0.0, 1.2, 1.0, 0.5, 0.0, 0.0, rng)
    assert nx[0] == pytest.approx(1.0 + 0.1 * 2.0)
    assert ny[0] == pytest.approx(3.0 - 0.1 * 2.0)
    assert (nx + ny)[0] == pytest.approx(4.0)


def test_user_relaxes_to_realised_value():
    p = Params(c=0.5, gam=0.0)
    rng = np.random.default_rng(0)
    nx = _update(np.array([2.0]), np.array([0.3]), np.array([True]), np.array([False]),
                 p, 1.0, 1.2, 1.0, 0.8, 0.0, 0.0, rng)
    assert nx[0] == pytest.approx(2.0 - 0.5 * (2.0 - 0.8))


def test_stationary_distribution_matches_fokker_planck():
    """With (almost) no purchases the non-users' values settle to the inverse-gamma law."""
    p = Params(lam=1e-9)
    s = fixed_signals(media=1.0, message=1.2)
    d = dsmc.simulate(p, s, N=40_000, mu_c=1000, T=3.0, init="message", seed=3, record_every=1.0)
    # mean of the inverse-gamma stationary law (r does not enter at rho = 0)
    shape, beta = mf.distribution_parameters(0.0, 0.4, 3.0, p, s)
    assert d["m1"].iloc[-1] == pytest.approx(invgamma.mean(shape, scale=beta), rel=0.01)


def test_stationary_tail_matches_hazard():
    """Share of agents above the price at equilibrium = mean-field hazard."""
    p = Params(lam=1e-9)
    s = fixed_signals(media=1.0, message=0.95)        # m1 below the price: a genuine tail
    rng = np.random.default_rng(4)
    v = dsmc.initial_values(40_000, p, s, rng, init="message")
    # run the collision kernel only, for 3 time units
    user = np.zeros(v.size, dtype=bool)
    sd = np.sqrt(p.eta)
    for _ in range(3000):
        perm = rng.permutation(v.size)
        i, j = perm[: v.size // 2], perm[v.size // 2:]
        vi, vj = v[i], v[j]
        v[i] = _update(vi, vj, user[i], user[j], p, 1.0, 0.95, 1.0, 0.4, sd, sd, rng)
        v[j] = _update(vj, vi, user[j], user[i], p, 1.0, 0.95, 1.0, 0.4, sd, sd, rng)
    h = mf.adoption_hazard(0.0, 0.4, 0.0, p, s)
    assert (v > 1.0).mean() == pytest.approx(h, abs=0.01)
    shape, beta = mf.distribution_parameters(0.0, 0.4, 0.0, p, s)
    assert v.std() == pytest.approx(invgamma.std(shape, scale=beta), rel=0.05)


def test_reproducible_with_seed():
    p, s = Params(), Signals()
    a = dsmc.simulate(p, s, N=2000, mu_c=500, T=1.0, seed=7)
    b = dsmc.simulate(p, s, N=2000, mu_c=500, T=1.0, seed=7)
    assert a.equals(b)


def test_adoption_never_decreases():
    d = dsmc.simulate(Params(), Signals(), N=3000, mu_c=500, T=5.0, seed=1)
    assert (np.diff(d["rho"]) >= 0).all()


def test_rejects_collision_probability_above_one():
    with pytest.raises(ValueError):
        dsmc.simulate(Params(), Signals(), N=100, mu_c=1000, dt=0.01, T=0.1)


def test_converges_to_mean_field_as_relaxation_speeds_up():
    """Faster relaxation (more collisions, same per-collision coefficients) brings
    the DSMC closer to the time-scale-separated mean-field solution."""
    p, s = Params(), Signals()
    ref = mf.simulate(p, s, T=4.0)
    slow = dsmc.simulate(p, s, N=5000, mu_c=500, T=4.0, seed=0)
    fast = dsmc.simulate(p, s, N=5000, mu_c=4000, T=4.0, seed=0)
    e_slow, e_fast = trajectory_error(ref, slow), trajectory_error(ref, fast)
    assert e_fast["rho"] < 0.6 * e_slow["rho"]
    assert e_fast["m1"] < 0.6 * e_slow["m1"]


def test_simulation_loop_conserves_mean_of_non_user_compromise():
    """With no media, no noise and no purchases, pairwise compromise among non-users
    conserves the population mean exactly. Catches a loop that updates the second
    agent of a pair from its partner's already-updated value."""
    N = 2000
    p = Params(gam=0.0, lam=1e-9, a1=0.05)
    s = fixed_signals(media=0.0, message=1.5)
    v0 = np.random.default_rng(5).uniform(0.5, 1.5, N)
    d = dsmc.simulate(p, s, N=N, mu_c=1000, T=0.5, init=v0, seed=2, record_every=0.1)
    assert d["rho"].iloc[-1] == 0.0
    assert d["m_all"].iloc[-1] == pytest.approx(v0.mean(), abs=1e-10)
    assert d["m_all"].iloc[0] == pytest.approx(v0.mean(), abs=1e-12)
