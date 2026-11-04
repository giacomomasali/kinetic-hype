"""Produce every DSMC result used by the notebooks (CSV files in results/).

    python scripts/run_experiments.py           # about 30 minutes on 2 cores
    python scripts/run_experiments.py --quick --out test_results   # smaller runs, a few minutes

Experiments
  speed   DSMC vs mean field as relaxation speeds up (interaction rate mu_c
          raised, per-collision coefficients fixed), N = 10^4, several seeds.
  size    Finite-population fluctuations: spread across seeds as N grows.
  fixed   Constant media and message, agents starting from a low prior:
          does a hype cycle appear, and for which relaxation speeds?
  noise   The fixed-media runs with and without self-reasoning noise (gam = 0).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from kinetic_hype import Params, Signals, dsmc, fixed_signals, hype_features, meanfield, trajectory_error

OUT = Path(__file__).resolve().parent.parent / "results"
# mu_c: interaction (collision) rate per agent per unit time, as in the paper


def _speed_job(args):
    mu_c, seed, N, T = args
    p, s = Params(), Signals()
    d = dsmc.simulate(p, s, N=N, mu_c=mu_c, T=T, seed=seed)
    d.insert(0, "mu_c", mu_c)
    d.insert(1, "seed", seed)
    return d


def _size_job(args):
    N, seed, mu_c, T = args
    d = dsmc.simulate(Params(), Signals(), N=N, mu_c=mu_c, T=T, seed=seed)
    d.insert(0, "N", N)
    d.insert(1, "seed", seed)
    return d


def _fixed_job(args):
    mu_c, v0, seed, N, T = args
    s = fixed_signals(media=1.0, message=1.5, investment_slope=0.1)
    d = dsmc.simulate(Params(), s, N=N, mu_c=mu_c, T=T, seed=seed, init=v0)
    d.insert(0, "mu_c", mu_c)
    d.insert(1, "v0", v0)
    d.insert(2, "seed", seed)
    return d


def _noise_job(args):
    gam, mu_c, seed, N, T = args
    s = fixed_signals(media=1.0, message=1.5, investment_slope=0.1)
    d = dsmc.simulate(Params(gam=gam), s, N=N, mu_c=mu_c, T=T, seed=seed, init=0.3)
    d.insert(0, "gam", gam)
    d.insert(1, "mu_c", mu_c)
    d.insert(2, "seed", seed)
    return d


def run(jobs, fn, n_jobs):
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(fn, jobs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", choices=["speed", "size", "fixed", "noise"])
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: results/); use another folder for test runs")
    a = ap.parse_args()
    global OUT
    if a.out is not None:
        OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    q = a.quick

    if a.only in (None, "speed"):
        t0 = time.time()
        rates = [500, 1000, 2000, 4000] if q else [500, 1000, 2000, 4000, 8000, 16_000, 32_000]
        seeds = [0] if q else [0, 1, 2]
        N, T = (4000, 6.0) if q else (10_000, 10.0)
        traj = pd.concat(run([(m, sd, N, T) for m in rates for sd in seeds], _speed_job, a.jobs))
        ref = meanfield.simulate(Params(), Signals(), T=T)
        ref.to_csv(OUT / "meanfield_default.csv", index=False)
        rows = []
        for (m, sd), d in traj.groupby(["mu_c", "seed"]):
            e = trajectory_error(ref, d)
            e_late = trajectory_error(ref, d, t_from=1.0)
            rows.append({"mu_c": m, "seed": sd, **{f"rms_{c}": v for c, v in e.items()},
                         **{f"rms_{c}_after_t1": v for c, v in e_late.items()}})
        pd.DataFrame(rows).to_csv(OUT / "convergence_speed.csv", index=False)
        traj[traj["seed"] == 0].to_csv(OUT / "trajectories_speed.csv", index=False)
        print(f"speed: {time.time() - t0:.0f}s")

    if a.only in (None, "size"):
        t0 = time.time()
        Ns = [500, 2000, 8000] if q else [500, 2000, 8000, 32_000]
        seeds = range(3 if q else 6)
        mu_c, T = (2000, 5.0) if q else (8000, 5.0)
        traj = pd.concat(run([(N, sd, mu_c, T) for N in Ns for sd in seeds], _size_job, a.jobs))
        traj.to_csv(OUT / "trajectories_size.csv", index=False)
        print(f"size: {time.time() - t0:.0f}s")

    if a.only in (None, "fixed"):
        t0 = time.time()
        rates = [250, 1000, 4000] if q else [125, 250, 500, 1000, 2000, 4000, 8000]
        N, T = (4000, 20.0) if q else (10_000, 20.0)
        traj = pd.concat(run([(m, 0.3, 0, N, T) for m in rates], _fixed_job, a.jobs))
        traj.to_csv(OUT / "trajectories_fixed.csv", index=False)
        s = fixed_signals(media=1.0, message=1.5, investment_slope=0.1)
        meanfield.simulate(Params(), s, T=T).to_csv(OUT / "meanfield_fixed.csv", index=False)
        rows = []
        for m, d in traj.groupby("mu_c"):
            f = hype_features(d, smooth=5)
            rows.append({"mu_c": m, "t_peak": f.t_peak, "peak": f.peak, "t_trough": f.t_trough,
                         "trough": f.trough, "final": f.final, "drop": f.drop,
                         "recovery": f.recovery, "hype_cycle": f.is_hype_cycle(t_min=0.1)})
        pd.DataFrame(rows).to_csv(OUT / "fixed_signals_features.csv", index=False)
        print(f"fixed: {time.time() - t0:.0f}s")

    if a.only in (None, "noise"):
        t0 = time.time()
        N, T = (4000, 20.0) if q else (10_000, 20.0)
        traj = pd.concat(run([(g, m, 0, N, T) for g in (0.0, 0.8) for m in (1000, 4000)], _noise_job, a.jobs))
        rows = []
        for (g, m), d in traj.groupby(["gam", "mu_c"]):
            f = hype_features(d, smooth=5)
            rows.append({"gam": g, "mu_c": m, "t_peak": f.t_peak, "peak": f.peak, "trough": f.trough,
                         "final": f.final, "drop": f.drop, "recovery": f.recovery,
                         "hype_cycle": f.is_hype_cycle(t_min=0.1), "rho_T": d["rho"].iloc[-1]})
        pd.DataFrame(rows).to_csv(OUT / "fixed_signals_noise.csv", index=False)
        print(f"noise: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
