"""Global sensitivity analysis of the mean-field system (results in results/).

    python scripts/run_sensitivity.py                 # Sobol design, n = 2048 (about 15 min on 2 cores)
    python scripts/run_sensitivity.py --quick --out test_results   # n = 256, kept out of results/
    python scripts/run_sensitivity.py --maps          # two-factor maps around the baseline

Factors and ranges: kinetic_hype.sensitivity.FACTORS. Outputs:
  sensitivity_runs.csv.gz     every run of the Saltelli design (factor values and outcomes)
  sensitivity_indices.csv     first-order and total Sobol indices with bootstrap 95% intervals
  sensitivity_map_<x>_<y>.csv outcomes on a grid of two factors, the others at the baseline
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from kinetic_hype import sensitivity as sa

OUT = Path(__file__).resolve().parent.parent / "results"
MAP_PAIRS = (("w0", "r_inf"), ("alpha2", "r_inf"), ("media_width2", "k1"))
INDICATORS = {"hype_cycle": "hype cycle (0/1)", "any_overshoot": "overshoot, with or without recovery (0/1)",
              "drop": "peak minus trough of m1", "rho_T": "adoption at T = 30"}


def _eval(x):
    return sa.evaluate(x)


def run_points(points, jobs):
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(_eval, points, chunksize=64))


def sobol(n, jobs, seed=0):
    design = sa.saltelli_design(n, seed=seed)
    blocks = ["A", "B"] + [f"AB{i}" for i in range(len(sa.FACTORS))]
    frames = []
    for blk in blocks:
        pts = sa.to_points(design[blk])
        df = pd.concat([pd.DataFrame(pts), pd.DataFrame(run_points(pts, jobs))], axis=1)
        df.insert(0, "block", blk)
        df.insert(1, "row", np.arange(n))
        frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    runs["any_overshoot"] = (runs["pattern"] >= 1).astype(float)
    runs.to_csv(OUT / "sensitivity_runs.csv.gz", index=False)

    rows = []
    for out in INDICATORS:
        get = lambda blk: runs.loc[runs.block == blk, out].to_numpy()
        res = sa.sobol_indices(get("A"), get("B"),
                               np.array([get(f"AB{i}") for i in range(len(sa.FACTORS))]), seed=seed)
        for j, f in enumerate(sa.FACTORS):
            rows.append({"outcome": out, "factor": f.name, **{k: res[k][j] for k in
                         ("S1", "S1_lo", "S1_hi", "ST", "ST_lo", "ST_hi")}, "n_used": res["n_used"]})
    pd.DataFrame(rows).to_csv(OUT / "sensitivity_indices.csv", index=False)


def maps(n_grid, jobs):
    fac = {f.name: f for f in sa.FACTORS}
    for a, b in MAP_PAIRS:
        ga = fac[a].scale(np.linspace(0, 1, n_grid))
        gb = fac[b].scale(np.linspace(0, 1, n_grid))
        pts = [{**sa.BASELINE, a: va, b: vb} for va, vb in itertools.product(ga, gb)]
        df = pd.concat([pd.DataFrame(pts)[[a, b]], pd.DataFrame(run_points(pts, jobs))], axis=1)
        df.to_csv(OUT / f"sensitivity_map_{a}_{b}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--maps", action="store_true", help="only the two-factor maps")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: results/); use another folder for test runs")
    a = ap.parse_args()
    global OUT
    if a.out is not None:
        OUT = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not a.maps:
        sobol(256 if a.quick else 2048, a.jobs)
        print(f"sobol: {time.time() - t0:.0f}s")
    t0 = time.time()
    maps(15 if a.quick else 41, a.jobs)
    print(f"maps: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
