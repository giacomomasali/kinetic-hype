"""Measuring hype cycles in trajectories and comparing solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HypeFeatures:
    """Peak, trough and recovery of an expectations curve."""

    t_peak: float
    peak: float
    t_trough: float
    trough: float
    final: float

    @property
    def drop(self) -> float:
        """Peak minus trough (size of the disillusionment)."""
        return self.peak - self.trough

    @property
    def recovery(self) -> float:
        """Final value minus trough."""
        return self.final - self.trough

    def is_hype_cycle(self, min_drop: float = 0.02, min_recovery: float = 0.01,
                      t_min: float = 0.0) -> bool:
        """Rise to an interior peak, fall by at least ``min_drop``, then recover.

        ``t_min`` excludes peaks at the very start (a curve that only falls is
        not a hype cycle).
        """
        return (self.t_peak > t_min and self.t_trough > self.t_peak
                and self.drop >= min_drop and self.recovery >= min_recovery)


def hype_features(df: pd.DataFrame, col: str = "m1", smooth: int = 1) -> HypeFeatures:
    """Global peak of ``col``, lowest point after it, and the final value.

    ``smooth`` > 1 applies a centred moving average first (use it for DSMC
    output, where sampling noise can create spurious extrema).
    """
    y = df[col].to_numpy(dtype=float)
    if smooth > 1:
        y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    t = df["t"].to_numpy(dtype=float)
    k_peak = int(np.nanargmax(y))
    k_trough = k_peak + int(np.nanargmin(y[k_peak:]))
    return HypeFeatures(t[k_peak], y[k_peak], t[k_trough], y[k_trough], y[~np.isnan(y)][-1])


@dataclass(frozen=True)
class LocalFall:
    """Largest fall of an expectations curve from any local peak.

    The trough is the lowest point between the peak and the first time the
    curve rises above that peak again (or the end of the run). ``exceeded`` is
    True when the curve does rise above the peak later, i.e. the peak is not
    the global maximum and expectations end the fall above where they started it.
    """

    t_peak: float
    peak: float
    t_trough: float
    trough: float
    exceeded: bool

    @property
    def fall(self) -> float:
        return self.peak - self.trough


def largest_fall(df: pd.DataFrame, col: str = "m1", t_min: float = 0.1, smooth: int = 1) -> LocalFall:
    """Largest peak-to-trough fall of ``col`` over all local peaks after ``t_min``.

    ``hype_features`` measures the fall from the *global* peak only, so it
    misses a temporary dip after an early peak that the curve later surpasses
    (expectations follow the media message, fall towards a still-immature
    performance as media attention fades, then rise with performance).
    """
    y = df[col].to_numpy(dtype=float)
    if smooth > 1:
        y = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    t = df["t"].to_numpy(dtype=float)
    n = y.size
    best = LocalFall(np.nan, np.nan, np.nan, np.nan, False)
    best_fall = 0.0
    for k in np.flatnonzero((t[1:-1] > t_min) & (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])) + 1:
        above = np.flatnonzero(y[k + 1:] > y[k])
        end = k + 1 + above[0] if above.size else n
        j = k + int(np.nanargmin(y[k:end]))
        if y[k] - y[j] > best_fall:
            best_fall = y[k] - y[j]
            best = LocalFall(t[k], y[k], t[j], y[j], bool(above.size))
    return best


def trajectory_error(reference: pd.DataFrame, other: pd.DataFrame,
                     cols=("rho", "m1", "r"), t_from: float = 0.0) -> dict[str, float]:
    """Root-mean-square difference between two trajectories on other's time grid."""
    out = {}
    mask = other["t"].to_numpy() >= t_from
    t = other["t"].to_numpy()[mask]
    for c in cols:
        ref = np.interp(t, reference["t"], reference[c])
        diff = ref - other[c].to_numpy()[mask]
        out[c] = float(np.sqrt(np.nanmean(diff ** 2)))
    return out
