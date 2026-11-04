"""Mean-field (slow time-scale) dynamics.

On the fast time scale non-users' perceived values relax to a stationary
distribution; on the slow scale adoption, cumulative expectations and the
realised value evolve. With mu = mu_m(t), alpha_i = a_i / b and rho = share of
users (rho1 = 1 - rho):

  Mean perceived value of non-users (fast-scale fixed point)

      m1 = P (a2 rho r + mu b w) / (a2 rho + mu b)

  Stationary density of non-users' values (Fokker-Planck with drift
  -[a1 rho1 (v - m1) + a2 rho (v - P r) + mu b (v - P w)] and noise variance
  gam b v^2, both per collision):

      f(v) ∝ v^{-(2 + kappa)} exp(-beta / v)                     (inverse gamma)
      kappa = 2 (mu + alpha1 rho1 + alpha2 rho) / gam
      beta  = (2 / gam) (alpha1 rho1 m1 + alpha2 rho P r + mu P w)

  Adoption hazard = P(V > P) = regularised lower incomplete gamma

      H = gammainc(1 + kappa, beta / P)

  Dynamics

      d rho / dt = lam (1 - rho) H,      d M / dt = m1,      r = r(M, rho, t)

Holding the media frequency at 1 in kappa and beta (adoption_hazard_unit_media)
overstates adoption once media attention fades; the paper uses this comparison
to quantify the role of mu. As gam -> 0 the hazard becomes the step
function 1[m1 > P] (deterministic case).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import gammainc
from scipy.stats import invgamma

from .model import Params, Signals, realised_value

_EPS = 1e-300


def mean_nonusers(rho: float, r: float, t: float, p: Params, s: Signals) -> float:
    """m1: fast-scale fixed point of the non-users' mean perceived value.

    Undefined when nobody has adopted and the media frequency is zero (then
    interactions among non-users conserve their mean). A tiny floor keeps the
    formula finite; avoid media signals that are exactly zero at rho = 0.
    """
    mu, w, P = s.media(t), s.message(t), s.price(t)
    num = p.a2 * rho * r + mu * p.b * w
    den = p.a2 * rho + mu * p.b
    return P * num / max(den, _EPS)


def distribution_parameters(rho: float, r: float, t: float, p: Params, s: Signals):
    """(shape, beta) of the inverse-gamma stationary density of non-users.

    shape = 1 + kappa, scale = beta. Requires gam > 0.
    """
    if p.gam <= 0:
        raise ValueError("the stationary density is a point mass when gam = 0")
    mu, w, P = s.media(t), s.message(t), s.price(t)
    m1 = mean_nonusers(rho, r, t, p, s)
    rho1 = 1.0 - rho
    kappa = 2.0 * (mu + p.alpha1 * rho1 + p.alpha2 * rho) / p.gam
    beta = (2.0 / p.gam) * (p.alpha1 * rho1 * m1 + p.alpha2 * rho * P * r + mu * P * w)
    return 1.0 + kappa, beta


def stationary_density(v, rho: float, r: float, t: float, p: Params, s: Signals):
    """Normalised stationary density of non-users' perceived values at v."""
    shape, beta = distribution_parameters(rho, r, t, p, s)
    return invgamma.pdf(v, a=shape, scale=beta)


def adoption_hazard(rho: float, r: float, t: float, p: Params, s: Signals) -> float:
    """Share of non-users whose perceived value exceeds the price."""
    P = s.price(t)
    if p.gam == 0:
        return float(mean_nonusers(rho, r, t, p, s) > P)
    shape, beta = distribution_parameters(rho, r, t, p, s)
    return float(gammainc(shape, beta / P))


def adoption_hazard_unit_media(rho: float, r: float, t: float, p: Params, s: Signals) -> float:
    """Hazard with the media frequency held at 1 in kappa and beta.

    For comparison only: it shows how much the media frequency matters for
    adoption once media attention fades. Not used by default.
    """
    w, P = s.message(t), s.price(t)
    m1 = mean_nonusers(rho, r, t, p, s)
    rho1 = 1.0 - rho
    shape = 1.0 + 2.0 * (1.0 + p.alpha1 * rho1 + p.alpha2 * rho) / p.gam
    arg = (2.0 / p.gam) * (p.alpha1 * rho1 * m1 / P + p.alpha2 * rho * r + w)
    return float(gammainc(shape, arg))


def simulate(p: Params, s: Signals, T: float = 20.0, n_points: int = 2001,
             rho0: float = 0.0, max_step: float | None = None,
             hazard: str = "derived") -> pd.DataFrame:
    """Integrate the mean-field system on [0, T].

    hazard: "derived" (default) or "unit_media" (media frequency held at 1 in
    the hazard, for comparison only).

    Returns a DataFrame with columns t, rho, m1, r, investment, media, message, price.
    """
    if hazard not in ("derived", "unit_media"):
        raise ValueError("hazard must be 'derived' or 'unit_media'")
    hazard_fn = adoption_hazard if hazard == "derived" or p.gam == 0 else adoption_hazard_unit_media
    def rhs(t, y):
        rho, M = min(max(y[0], 0.0), 1.0), y[1]
        r = realised_value(M, rho, t, p, s)
        m1 = mean_nonusers(rho, r, t, p, s)
        h = hazard_fn(rho, r, t, p, s)
        return [p.lam * (1.0 - rho) * h, m1]

    t_eval = np.linspace(0.0, T, n_points)
    if p.gam == 0:
        # Deterministic case: the hazard is a step function of the state, which
        # stalls adaptive solvers at the switch. Use fixed-step RK4 instead.
        ts, ys = _rk4_fixed(rhs, [rho0, 0.0], T, max_step or T / 20_000, t_eval)
    else:
        sol = solve_ivp(rhs, (0.0, T), [rho0, 0.0], t_eval=t_eval, method="RK45",
                        rtol=1e-8, atol=1e-10, max_step=max_step or T / 400)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts, ys = sol.t, sol.y
    rows = []
    for t, rho, M in zip(ts, ys[0], ys[1]):
        r = realised_value(M, rho, t, p, s)
        rows.append({
            "t": t, "rho": rho, "m1": mean_nonusers(rho, r, t, p, s), "r": r,
            "investment": p.k1 * M + p.k2 * rho + s.investment(t),
            "media": s.media(t), "message": s.message(t), "price": s.price(t),
        })
    return pd.DataFrame(rows)


def _rk4_fixed(rhs, y0, T, h, t_eval):
    """Classical RK4 with constant step h, sampled at t_eval by linear interpolation."""
    n = int(np.ceil(T / h))
    h = T / n
    ts = np.linspace(0.0, T, n + 1)
    ys = np.empty((len(y0), n + 1))
    y = np.asarray(y0, dtype=float)
    ys[:, 0] = y
    for k in range(n):
        t = ts[k]
        k1 = np.asarray(rhs(t, y))
        k2 = np.asarray(rhs(t + h / 2, y + h / 2 * k1))
        k3 = np.asarray(rhs(t + h / 2, y + h / 2 * k2))
        k4 = np.asarray(rhs(t + h, y + h * k3))
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys[:, k + 1] = y
    out = np.vstack([np.interp(t_eval, ts, row) for row in ys])
    return t_eval, out
