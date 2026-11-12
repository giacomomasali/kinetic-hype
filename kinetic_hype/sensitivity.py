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
from .analysis import hype_features, largest_fall
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
OUTCOMES = ("pattern", "hype_cycle", "ends_above_peak", "drop", "recovery", "overshoot", "max_fall",
            "rho_T", "t_peak", "r_media_peak")
MEDIA_PEAK = 2.0          # time at which media attention peaks in every run (centre of the Gaussian bump)
PATTERNS = {0: "no fall", 1: "fall, no recovery", 2: "hype cycle"}


def build(x: dict) -> tuple[Params, Signals]:
    """Params and Signals for one point; b is fixed at 1e-3 so a_i = alpha_i * b."""
    b = 1e-3
    p = Params(a1=x["alpha1"] * b, a2=x["alpha2"] * b, b=b, gam=x["gam"], lam=x["lam"],
               r_inf=x["r_inf"], nu=x["nu"], k1=x["k1"], k2=x["k2"])
    s = Signals(media=gaussian(MEDIA_PEAK, x["media_width2"]),
                message=rise_then_flat(x["w0"], center=2.0, width2=2.0),
                investment=linear(x["inv_slope"]))
    return p, s


def evaluate(x: dict, T: float = 30.0, n_points: int = 601) -> dict:
    """Outcomes of one mean-field run.

    pattern     classification of m1(t), with min_drop = 0.02 and min_recovery = 0.01:
                0: no fall (m1 never falls by min_drop or more from any peak after t = 0.1);
                1: fall without recovery (m1 falls by min_drop or more from its global
                   peak and settles: a hype-disappointment curve);
                2: hype cycle (m1 falls by min_drop or more from a peak after t = 0.1 and
                   then recovers by min_recovery or more). The peak may be the global
                   maximum (the trough lies below the final level) or a local one that
                   expectations later surpass (they end above the early peak).
    hype_cycle       1 if pattern == 2, else 0
    ends_above_peak  1 if a hype cycle in which expectations later rise above the
                     peak they fell from, else 0
    drop        global peak minus following trough of m1
    recovery    final value minus trough of m1 after the global peak
    overshoot   global peak minus final value of m1
    max_fall    largest fall of m1 from any peak after t = 0.1 (analysis.largest_fall)
    r_media_peak     realised performance r at the media peak (t = MEDIA_PEAK); an outcome of
                     the run, not an input, since investment depends on expectations
    """
    p, s = build(x)
    try:
        d = meanfield.simulate(p, s, T=T, n_points=n_points)
    except Exception:  # solver failure: recorded as missing
        return {k: np.nan for k in OUTCOMES}
    f = hype_features(d)
    loc = largest_fall(d, t_min=0.1)
    global_fall = f.drop if f.t_peak > 0.1 else 0.0
    max_fall = max(loc.fall, global_fall) if np.isfinite(loc.fall) else global_fall
    global_cycle = f.is_hype_cycle(t_min=0.1)
    fall_no_recovery = f.t_peak > 0.1 and f.drop >= 0.02 and not global_cycle
    # a fall from a local peak that expectations later surpass: recovery exceeds the fall
    above = bool(np.isfinite(loc.fall) and loc.fall >= 0.02 and loc.exceeded)
    cycle = global_cycle or (above and not fall_no_recovery)
    pattern = 2 if cycle else (1 if fall_no_recovery else 0)
    return {"pattern": float(pattern), "hype_cycle": float(cycle),
            "ends_above_peak": float(cycle and not global_cycle), "drop": f.drop, "recovery": f.recovery,
            "overshoot": f.peak - f.final, "max_fall": float(max_fall), "rho_T": float(d.rho.iloc[-1]),
            "t_peak": f.t_peak, "r_media_peak": float(np.interp(MEDIA_PEAK, d.t, d.r))}


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
