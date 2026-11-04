"""Direct Simulation Monte Carlo of the agent model (Nanbu-Babovsky scheme).

Each time step of length dt:

1. Macroscopic quantities are measured: share of users rho, mean perceived
   value of non-users m1, cumulative expectations M (Riemann sum of m1), and
   the realised value r.
2. Agents are shuffled and paired without replacement (Babovsky's variant of
   Nanbu's scheme). Each pair collides with probability mu_c * dt, and both
   agents are updated from their *pre-collision* values.
3. Each non-user gets a purchase opportunity with probability lam * dt and
   adopts if its perceived value exceeds the price.

The collision rate mu_c and the per-collision coefficients in Params set how
fast perceived values relax compared with purchases. The mean-field model is
the limit of many agents (N -> inf) and fast relaxation compared with
purchases (time-scale separation). To approach it, raise mu_c with the
per-collision coefficients fixed: perceived values then relax faster relative
to purchases (see notebooks/03_dsmc_convergence.ipynb). Raising mu_c while dividing
the coefficients by the same factor would keep the relaxation rate fixed and
only make each jump smaller, without bringing the DSMC closer to the mean field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import invgamma

from .meanfield import distribution_parameters, mean_nonusers
from .model import Params, Signals, realised_value


def initial_values(N: int, p: Params, s: Signals, rng: np.random.Generator,
                   init: str | float | np.ndarray = "stationary") -> np.ndarray:
    """Perceived values at t = 0 (everyone a non-user).

    "stationary": draw from the fast-scale stationary density at t = 0, which is
    what the mean-field model assumes. "message": everyone at P(0) w(0).
    A number v0: everyone starts at v0 (a common prior before any media).
    An array of N positive values: used as given (copied).
    """
    if isinstance(init, np.ndarray):
        if init.shape != (N,) or np.any(init <= 0):
            raise ValueError("an initial array must have N positive values")
        return init.astype(float).copy()
    if not isinstance(init, str):
        if init <= 0:
            raise ValueError("initial perceived value must be positive")
        return np.full(N, float(init))
    if init == "message" or p.gam == 0:
        if init == "stationary":
            r0 = realised_value(0.0, 0.0, 0.0, p, s)
            return np.full(N, mean_nonusers(0.0, r0, 0.0, p, s))
        return np.full(N, s.price(0.0) * s.message(0.0))
    if init == "stationary":
        r0 = realised_value(0.0, 0.0, 0.0, p, s)
        shape, beta = distribution_parameters(0.0, r0, 0.0, p, s)
        return invgamma.rvs(a=shape, scale=beta, size=N, random_state=rng)
    raise ValueError("init must be 'stationary' or 'message'")


def simulate(p: Params, s: Signals, N: int = 10_000, mu_c: float = 1000.0,
             T: float = 20.0, dt: float | None = None, record_every: float = 0.05,
             seed: int | None = 0, init: str | float | np.ndarray = "stationary") -> pd.DataFrame:
    """Run the DSMC. Returns a DataFrame with the same columns as meanfield.simulate.

    dt defaults to 1 / mu_c, so every pair collides each step. The collision
    probability mu_c * dt must not exceed 1.
    """
    if dt is None:
        dt = 1.0 / mu_c
    p_coll = mu_c * dt
    if p_coll > 1 + 1e-12:
        raise ValueError(f"mu_c * dt = {p_coll:.3g} > 1: reduce dt")
    if not 0 < p.lam * dt <= 1:
        raise ValueError("lam * dt must be in (0, 1]")

    rng = np.random.default_rng(seed)
    v = initial_values(N, p, s, rng, init).astype(float)
    user = np.zeros(N, dtype=bool)
    sd_nonusers, sd_users = np.sqrt(p.eta), np.sqrt(p.users_noise)
    n_steps = int(round(T / dt))
    every = max(1, int(round(record_every / dt)))
    M = 0.0
    clipped = 0
    rows = []

    for k in range(n_steps + 1):
        t = k * dt
        rho = user.mean()
        nonusers = ~user
        m1 = v[nonusers].mean() if nonusers.any() else np.nan
        r = realised_value(M, rho, t, p, s)
        mu, w, P = s.media(t), s.message(t), s.price(t)
        if k % every == 0:
            rows.append({"t": t, "rho": rho, "m1": m1, "r": r,
                         "m_all": v.mean(),
                         "m_users": v[user].mean() if user.any() else np.nan,
                         "investment": p.k1 * M + p.k2 * rho + s.investment(t),
                         "media": mu, "message": w, "price": P})
        if k == n_steps:
            break

        # --- binary collisions (pairs without replacement) ---------------------
        perm = rng.permutation(N)
        half = N // 2
        i, j = perm[:half], perm[half:2 * half]
        hit = rng.random(half) < p_coll
        i, j = i[hit], j[hit]
        vi, vj = v[i], v[j]                      # pre-collision values (copies)
        ui, uj = user[i], user[j]
        v[i] = _update(vi, vj, ui, uj, p, mu, w, P, r, sd_nonusers, sd_users, rng)
        v[j] = _update(vj, vi, uj, ui, p, mu, w, P, r, sd_nonusers, sd_users, rng)
        neg = v <= 0
        if neg.any():                            # multiplicative noise can overshoot zero
            clipped += int(neg.sum())
            v[neg] = 1e-12

        # --- purchases -----------------------------------------------------------
        buy = (~user) & (rng.random(N) < p.lam * dt) & (v > P)
        user |= buy

        if not np.isnan(m1):
            M += m1 * dt

    out = pd.DataFrame(rows)
    out.attrs.update(N=N, mu_c=mu_c, dt=dt, seed=seed, clipped=clipped, init=init)
    return out


def _update(x, y, ux, uy, p, mu, w, P, r, sd_n, sd_u, rng):
    """New values for agents x after colliding with partners y (pre-collision values)."""
    noise = rng.standard_normal(x.size)
    a = np.where(uy, p.a2, p.a1)
    nonuser_new = x - a * (x - y) - mu * p.b * (x - P * w) + x * sd_n * noise
    user_new = x - p.c * (x - P * r) + x * sd_u * noise
    return np.where(ux, user_new, nonuser_new)
