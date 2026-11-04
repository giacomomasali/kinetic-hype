"""Model definition shared by the mean-field solver and the DSMC.

Agents hold a perceived value v > 0 of an innovation and an adoption status
(non-user or user). Perceived values change through binary interactions; each
collision applies the compromise rules below, with per-collision coefficients:

    non-user meeting a non-user  v' = v - a1 (v - v*) - mu_m(t) b (v - P w) + v xi
    non-user meeting a user      v' = v - a2 (v - v*) - mu_m(t) b (v - P w) + v xi
    user                         v' = v - c  (v - P r) + v zeta

with xi ~ N(0, gam b) and zeta ~ N(0, xi_users). A non-user who receives a
purchase opportunity (rate lam) adopts if v exceeds the price P(t).

The technology's realised value is a logistic function of investment,

    r = r_inf / (1 + exp(-nu (I - I0))),   I = k1 * int_0^t m1 ds + k2 rho + I_exo(t),

where m1 is the mean perceived value of non-users and rho the share of users.

Exogenous signals (functions of time): media frequency mu_m(t), media message
w(t), price P(t), exogenous investment I_exo(t).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Signal = Callable[[float], float]


@dataclass(frozen=True)
class Params:
    """Model parameters.

    a1, a2, b, c are per-collision coefficients and gam scales the non-users'
    noise variance (eta = gam * b per collision). The mean-field model depends
    on them only through alpha_i = a_i / b and gam; the absolute values and the
    collision rate matter only in the DSMC.
    """

    a1: float = 0.01      # non-user <- non-user influence
    a2: float = 0.015     # non-user <- user influence
    b: float = 0.001      # non-user <- media influence
    gam: float = 0.8      # noise: non-users' variance per collision is gam * b
    lam: float = 0.8      # purchase opportunities per unit time
    r_inf: float = 0.8    # ceiling of realised value
    nu: float = 1.0       # steepness of the performance S-curve
    k1: float = 0.5       # weight of cumulative expectations in investment
    k2: float = 0.5       # weight of adoption in investment
    I0: float = 0.0       # investment at the S-curve midpoint
    c: float = 1.0        # users' relaxation towards P r per collision (DSMC only)
    xi_users: float | None = None  # users' noise variance per collision; default gam * b

    def __post_init__(self):
        if self.b <= 0:
            raise ValueError("b must be positive (the model is written in terms of a_i / b)")
        for name in ("a1", "a2", "gam", "lam", "r_inf", "nu", "c"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def alpha1(self) -> float:
        return self.a1 / self.b

    @property
    def alpha2(self) -> float:
        return self.a2 / self.b

    @property
    def eta(self) -> float:
        """Non-users' noise variance per collision."""
        return self.gam * self.b

    @property
    def users_noise(self) -> float:
        return self.eta if self.xi_users is None else self.xi_users



# --- exogenous signal helpers ------------------------------------------------------

def constant(value: float) -> Signal:
    return lambda t: float(value)


def gaussian(center: float, width2: float, height: float = 1.0, base: float = 0.0) -> Signal:
    """base + height * exp(-(t - center)^2 / width2)."""
    return lambda t: base + height * float(np.exp(-((t - center) ** 2) / width2))


def rise_then_flat(level: float, center: float, width2: float) -> Signal:
    """Gaussian rise to ``level`` at ``center``, constant afterwards."""
    def f(t):
        return level * float(np.exp(-((t - center) ** 2) / width2)) if t < center else float(level)
    return f


def linear(slope: float, intercept: float = 0.0) -> Signal:
    return lambda t: intercept + slope * float(t)


@dataclass(frozen=True)
class Signals:
    """Exogenous inputs as functions of time."""

    media: Signal = field(default_factory=lambda: gaussian(center=2.0, width2=2.0))
    message: Signal = field(default_factory=lambda: rise_then_flat(1.5, center=2.0, width2=2.0))
    price: Signal = field(default_factory=lambda: constant(1.0))
    investment: Signal = field(default_factory=lambda: linear(0.1))


def fixed_signals(media: float = 1.0, message: float = 1.5, price: float = 1.0,
                  investment_slope: float = 0.1) -> Signals:
    """Signals that do not change over time (except exogenous investment)."""
    return Signals(media=constant(media), message=constant(message), price=constant(price),
                   investment=linear(investment_slope))


# --- quantities shared by both solvers ----------------------------------------------

def realised_value(cum_m1: float, rho: float, t: float, p: Params, s: Signals) -> float:
    """r(t) from cumulative expectations, adoption and exogenous investment."""
    invest = p.k1 * cum_m1 + p.k2 * rho + s.investment(t)
    return p.r_inf / (1.0 + np.exp(-p.nu * (invest - p.I0)))
