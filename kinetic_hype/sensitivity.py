"""Global sensitivity analysis of the mean-field system.

A point of the parameter space is a dict of the factors in ``FACTORS``; it is
turned into ``Params`` and ``Signals`` by ``build``. ``evaluate`` runs the
mean-field solver and returns the outcomes of interest.

Variance-based (Sobol) indices use the Saltelli design: two independent
sample matrices A and B and, for each factor i, the matrix AB_i equal to A with
column i taken from B. With outputs f(A), f(B), f(AB_i):

    first-order  S_i  = mean(f(B) * (f(AB_i) - f(A))) / Var(Y)      (Saltelli et al. 2010)
    total        ST_i = mean((f(A) - f(AB_i))**2) / (2 Var(Y))     (Jansen 1999)

Reference: Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M.,
Tarantola, S. (2010). Variance based sensitivity analysis of model output.
Computer Physics Communications 181(2), 259-270.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from . import meanfield
from .analysis import hype_features
from .model import Params, Signals, gaussian, linear, rise_then_flat


@dataclass(frozen=True)
class Factor:
    name: str
    low: float
    high: float
    log: bool = False
    label: str = ""

    def scale(self, u: np.ndarray) -> np.ndarray:
        """Map u in [0, 1] to the factor's range (log-uniform if ``log``)."""
        if self.log:
            return np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low)))
        return self.low + u * (self.high - self.low)


# Ranges bracket the paper's baseline (Table 3) by roughly an order of magnitude
# where the parameter is a rate or weight, and by a plausible band otherwise.
FACTORS = (
    Factor("media_width2", 0.5, 32.0, True, r"media width $\sigma^2_\mu$"),
    Factor("w0", 0.8, 2.0, False, r"media message $w$"),
    Factor("r_inf", 0.6, 1.4, False, r"performance ceiling $r_\infty$"),
    Factor("k1", 0.1, 2.0, True, r"investment on expectations $k_1$"),
    Factor("k2", 0.1, 2.0, True, r"investment on adoption $k_2$"),
    Factor("inv_slope", 0.0, 0.5, False, r"exogenous investment slope"),
    Factor("gam", 0.2, 3.0, True, r"self-reasoning variance $\gamma$"),
    Factor("alpha1", 1.0, 50.0, True, r"non-user influence $\alpha_1=a_1/b$"),
    Factor("alpha2", 2.0, 100.0, True, r"user influence $\alpha_2=a_2/b$"),
    Factor("lam", 0.2, 3.0, True, r"purchase rate $\lambda$"),
    Factor("nu", 0.5, 3.0, True, r"performance steepness $\nu$"),
)
NAMES = tuple(f.name for f in FACTORS)
BASELINE = dict(media_width2=2.0, w0=1.5, r_inf=0.8, k1=0.5, k2=0.5, inv_slope=0.1, gam=0.8,
                alpha1=10.0, alpha2=15.0, lam=0.8, nu=1.0)
OUTCOMES = ("pattern", "hype_cycle", "drop", "recovery", "overshoot", "rho_T", "t_peak")
PATTERNS = {0: "no overshoot", 1: "overshoot, no recovery", 2: "hype cycle"}


def build(x: dict) -> tuple[Params, Signals]:
    """Params and Signals for one point; b is fixed at 1e-3 so a_i = alpha_i * b."""
    b = 1e-3
    p = Params(a1=x["alpha1"] * b, a2=x["alpha2"] * b, b=b, gam=x["gam"], lam=x["lam"],
               r_inf=x["r_inf"], nu=x["nu"], k1=x["k1"], k2=x["k2"])
    s = Signals(media=gaussian(2.0, x["media_width2"]),
                message=rise_then_flat(x["w0"], center=2.0, width2=2.0),
                investment=linear(x["inv_slope"]))
    return p, s


def evaluate(x: dict, T: float = 30.0, n_points: int = 601) -> dict:
    """Outcomes of one mean-field run.

    pattern     0: no overshoot (expectations never fall by 0.02 or more after
                   an interior peak); 1: overshoot without recovery (they fall
                   and settle, a hype-disappointment curve); 2: hype cycle
    hype_cycle  1 if pattern == 2 (analysis.HypeFeatures.is_hype_cycle), else 0
    drop        peak minus following trough of m1
    recovery    final value minus trough of m1
    overshoot   peak minus final value of m1
    rho_T       adoption at T
    t_peak      time of the peak of m1
    """
    p, s = build(x)
    try:
        d = meanfield.simulate(p, s, T=T, n_points=n_points)
    except Exception:  # solver failure: recorded as missing
        return {k: np.nan for k in OUTCOMES}
    f = hype_features(d)
    cycle = f.is_hype_cycle(t_min=0.1)
    overshoot = f.t_peak > 0.1 and f.drop >= 0.02
    pattern = 2 if cycle else (1 if overshoot else 0)
    return {"pattern": float(pattern), "hype_cycle": float(cycle), "drop": f.drop, "recovery": f.recovery,
            "overshoot": f.peak - f.final, "rho_T": float(d.rho.iloc[-1]), "t_peak": f.t_peak}


def to_points(u: np.ndarray) -> list[dict]:
    """Rows of a unit-cube matrix (n x d) to parameter dicts."""
    cols = [f.scale(u[:, j]) for j, f in enumerate(FACTORS)]
    return [dict(zip(NAMES, vals)) for vals in zip(*cols)]


def saltelli_design(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Unit-cube matrices A, B (n x d) from a scrambled Sobol sequence, and AB_i."""
    d = len(FACTORS)
    base = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
    A, B = base[:, :d], base[:, d:]
    design = {"A": A, "B": B}
    for i in range(d):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        design[f"AB{i}"] = ABi
    return design


def sobol_indices(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray,
                  n_boot: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
    """First-order and total indices with bootstrap 95% intervals.

    fA, fB: shape (n,); fAB: shape (d, n). Rows with any missing output are dropped.
    """
    ok = np.isfinite(fA) & np.isfinite(fB) & np.all(np.isfinite(fAB), axis=0)
    fA, fB, fAB = fA[ok], fB[ok], fAB[:, ok]

    def est(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        var = np.var(np.concatenate([a, b]), ddof=1)
        if var == 0:
            return np.full(ab.shape[0], np.nan), np.full(ab.shape[0], np.nan)
        s1 = np.mean(b * (ab - a), axis=1) / var
        st = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
        return s1, st

    n = fA.size
    s1, st = est(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    s1_b = np.array([b[0] for b in boot])
    st_b = np.array([b[1] for b in boot])
    return {"S1": s1, "ST": st,
            "S1_lo": np.nanpercentile(s1_b, 2.5, axis=0), "S1_hi": np.nanpercentile(s1_b, 97.5, axis=0),
            "ST_lo": np.nanpercentile(st_b, 2.5, axis=0), "ST_hi": np.nanpercentile(st_b, 97.5, axis=0),
            "n_used": n}
