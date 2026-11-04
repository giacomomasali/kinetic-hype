"""Figures and numbers for the working paper (written to paper/figures/).

    python scripts/run_experiments.py      # first: DSMC results in results/
    python scripts/make_paper_figures.py

Mean-field runs are computed here; DSMC trajectories are read from results/.
Every number quoted in the paper's Sections 4.2-4.4 is printed by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter

from kinetic_hype import Params, Signals, gaussian, hype_features, linear, meanfield
from kinetic_hype import sensitivity as sa
from kinetic_hype.plotting import QUANTITY, INK, INK_2, use_style

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"


def style():
    use_style()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.grid": False,
                         "font.size": 8, "axes.titlesize": 8.5, "lines.linewidth": 1.5,
                         "legend.fontsize": 7})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


LABEL = {"m1": r"Expectations $m_1(t)$", "rho": r"Adoption $\rho_2(t)$", "r": r"Realized performance $r(t)$"}


def three_curves(ax, d, labels=True):
    for c in ("m1", "rho", "r"):
        ax.plot(d.t, d[c], color=QUANTITY[c]["color"], label=LABEL[c] if labels else None)


def fig_noisy(numbers):
    p, s = Params(), Signals()
    d = meanfield.simulate(p, s, T=20)
    f = hype_features(d)
    det = meanfield.simulate(Params(gam=0.0), s, T=20)
    unit = meanfield.simulate(p, s, T=20, hazard="unit_media")
    fu = hype_features(unit)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    three_curves(ax, d)
    for t_, y_, lab, dy in [(f.t_peak, f.peak, "peak", 8), (f.t_trough, f.trough, "trough", -16)]:
        ax.plot(t_, y_, "o", color=QUANTITY["m1"]["color"], ms=6)
        ax.annotate(lab, (t_, y_), textcoords="offset points", xytext=(5, dy))
    ax.set_xlabel("time"); ax.set_ylabel("value"); ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); save(fig, "fig_noisy_hc")
    numbers["noisy"] = dict(t_peak=f.t_peak, peak=f.peak, t_trough=f.t_trough, trough=f.trough,
                            final=f.final, rho_T=d.rho.iloc[-1], rho_T_deterministic=det.rho.iloc[-1],
                            rho_T_unit_media_hazard=unit.rho.iloc[-1], peak_unit_media=fu.peak,
                            trough_unit_media=fu.trough)


def fig_sensitivity(numbers):
    panels = [
        ("Media attention width $\\sigma^2_\\mu$", "media width2",
         [(w2, Params(), Signals(media=gaussian(2.0, w2))) for w2 in (0.5, 2.0, 8.0, 32.0)],
         lambda v: f"$\\sigma^2_\\mu = {v:g}$"),
        ("Investment weight on expectations $k_1$", "k1",
         [(k, Params(k1=k), Signals()) for k in (0.25, 0.5, 1.0)], lambda v: f"$k_1 = {v:g}$"),
        ("Exogenous investment slope", "investment slope",
         [(sl, Params(), Signals(investment=linear(sl))) for sl in (0.0, 0.1, 0.3)],
         lambda v: f"$\\bar I(t) = {v:g}\\,t$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    rows = []
    for ax, (title, key, runs, lab) in zip(axes, panels):
        shades = plt.cm.Blues(np.linspace(0.35, 1.0, len(runs)))
        for (v, pp, ss), c in zip(runs, shades):
            d = meanfield.simulate(pp, ss, T=20)
            h = hype_features(d)
            ax.plot(d.t, d.m1, color=c, label=lab(v))
            rows.append(dict(varied=key, value=v, peak=h.peak, trough=h.trough, drop=h.drop,
                             hype_cycle=h.is_hype_cycle(t_min=0.1), rho_T=d.rho.iloc[-1]))
        ax.set_title(title); ax.set_xlabel("time"); ax.legend(loc="lower right", frameon=False, fontsize=6.5)
        ax.set_xlim(0, 12)
    axes[0].set_ylabel(r"expectations $m_1$")
    fig.tight_layout(); save(fig, "fig_sensitivity")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "sensitivity.csv", index=False)
    numbers["sensitivity"] = tab.round(4).to_dict(orient="records")


def fig_dsmc_match(numbers):
    ref = pd.read_csv(RES / "meanfield_default.csv")
    speed = pd.read_csv(RES / "trajectories_speed.csv")
    conv = pd.read_csv(RES / "convergence_speed.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, m in zip(axes[:2], (1000, 16_000)):
        three_curves(ax, ref, labels=(m == 16_000))
        d = speed[speed.mu_c == m]
        for j, c in enumerate(("m1", "rho", "r")):
            ax.plot(d.t, d[c], color=INK, ls=":", lw=1.6, label="DSMC" if (j == 0 and m == 16_000) else None)
        ax.set_title(f"$\\mu_c = {m:,}$".replace(",", "{,}"))
        ax.set_xlabel("time")
    axes[0].set_ylabel("value"); h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.36, -0.08))
    g = conv.groupby("mu_c").mean(numeric_only=True)
    for c in ("rho", "m1", "r"):
        axes[2].plot(g.index, g[f"rms_{c}_after_t1"], "o-", ms=4, color=QUANTITY[c]["color"],
                     label=LABEL[c])
    kk = np.array([g.index.min(), g.index.max()], dtype=float)
    y0 = g["rms_rho_after_t1"].iloc[0]
    axes[2].plot(kk, y0 * kk[0] / kk, color=INK_2, ls="--", lw=1, label=r"$\propto \mu_c^{-1}$")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel(r"interaction rate $\mu_c$"); axes[2].set_ylabel("RMS distance ($t>1$)")
    axes[2].legend(loc="lower left", frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, "fig_dsmc_match")

    slopes = {}
    sel = g.index >= 2000
    for c in ("rho", "m1", "r"):
        slopes[c] = float(np.polyfit(np.log(g.index[sel]), np.log(g[f"rms_{c}_after_t1"][sel]), 1)[0])
    p = Params()
    relax = 1000 * (p.b * ref.media + p.a2 * ref.rho) / p.lam
    numbers["dsmc_match"] = dict(
        rms_after_t1=g[[f"rms_{c}_after_t1" for c in ("rho", "m1", "r")]].round(4).reset_index().to_dict(orient="records"),
        slopes_after_t1_mu_c_ge_2000=slopes,
        relax_ratio_mu_c_1000_t0=float(relax.iloc[0]), relax_ratio_mu_c_1000_max=float(relax.max()),
        T=float(ref.t.iloc[-1]))
    for m in (1000, 16_000):
        h = hype_features(speed[speed.mu_c == m], smooth=5)
        numbers["dsmc_match"][f"features_mu_c_{m}"] = dict(t_peak=h.t_peak, peak=h.peak)
    h = hype_features(ref)
    numbers["dsmc_match"]["features_meanfield"] = dict(t_peak=h.t_peak, peak=h.peak)


def fig_fixed(numbers):
    traj = pd.read_csv(RES / "trajectories_fixed.csv")
    ref = pd.read_csv(RES / "meanfield_fixed.csv")
    feats = pd.read_csv(RES / "fixed_signals_features.csv")
    noise = pd.read_csv(RES / "fixed_signals_noise.csv")
    rates = sorted(traj.mu_c.unique())
    shades = plt.cm.Blues(np.linspace(0.3, 1.0, len(rates)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for m, c in zip(rates, shades):
        d = traj[traj.mu_c == m]
        lab = f"$\\mu_c = {m:g}$"
        axes[0].plot(d.t, d.m1, color=c, lw=1.4, label=lab)
        axes[1].plot(d.t, d.rho, color=c, lw=1.4)
    axes[0].plot(ref.t, ref.m1, color=INK, ls="--", lw=1.4, label="mean field")
    axes[1].plot(ref.t, ref.rho, color=INK, ls="--", lw=1.4)
    axes[0].set_xlim(0, 10)
    axes[0].set_ylabel(r"expectations $m_1$"); axes[1].set_ylabel(r"adoption $\rho_2$")
    for ax in axes:
        ax.set_xlabel("time")
    axes[0].legend(loc="lower right", frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); save(fig, "fig_fixed_media")
    h = hype_features(ref)
    numbers["fixed_media"] = dict(
        dsmc=feats.round(4).to_dict(orient="records"),
        meanfield=dict(t_peak=h.t_peak, peak=h.peak, t_trough=h.t_trough, trough=h.trough, final=h.final,
                       rho_T=ref.rho.iloc[-1]),
        noise=noise.round(4).to_dict(orient="records"),
        rho_T_dsmc={int(m): float(traj[traj.mu_c == m].rho.iloc[-1]) for m in rates})


def numbers_size(numbers):
    size = pd.read_csv(RES / "trajectories_size.csv")
    out = {}
    for N, d in size.groupby("N"):
        out[int(N)] = dict(sd_m1_time_avg=float(d.pivot(index="t", columns="seed", values="m1").std(axis=1).mean()),
                           sd_rho_final=float(d.pivot(index="t", columns="seed", values="rho").iloc[-1].std()))
    numbers["size"] = out


PATTERN_COLOURS = ["#d9d9d6", "#eb6834", "#2a78d6"]


def fig_global(numbers):
    """Global sensitivity: pattern shares against w - r_inf, and the (media width, k1) map."""
    runs = pd.read_csv(RES / "sensitivity_runs.csv.gz")
    idx = pd.read_csv(RES / "sensitivity_indices.csv")
    base = runs[runs.block.isin(["A", "B"])].copy()
    base["gap"] = base.w0 - base.r_inf

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    q = pd.qcut(base.gap, 12)
    tab = pd.crosstab(q, base.pattern, normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    mid = base.groupby(q, observed=True).gap.median().to_numpy()
    axes[0].stackplot(mid, tab[0], tab[1], tab[2], colors=PATTERN_COLOURS, labels=list(sa.PATTERNS.values()))
    axes[0].axvline(0, color=INK, lw=0.8, ls=":")
    axes[0].set_xlabel(r"media message minus performance ceiling, $w-r_\infty$")
    axes[0].set_ylabel("share of runs"); axes[0].set_ylim(0, 1); axes[0].set_xlim(mid[0], mid[-1])

    m = pd.read_csv(RES / "sensitivity_map_media_width2_k1.csv")
    xa, xb = np.sort(m.media_width2.unique()), np.sort(m.k1.unique())
    Z = m.pivot(index="k1", columns="media_width2", values="pattern").loc[xb, xa].to_numpy()
    D = m.pivot(index="k1", columns="media_width2", values="drop").loc[xb, xa].to_numpy()
    ax = axes[1]
    ax.pcolormesh(xa, xb, Z, cmap=ListedColormap(PATTERN_COLOURS), vmin=-0.5, vmax=2.5, shading="nearest", rasterized=True)
    cs = ax.contour(xa, xb, D, levels=[0.05, 0.1, 0.15, 0.2, 0.25], colors=INK, linewidths=0.6)
    ax.clabel(cs, fontsize=6, fmt="%.2f")
    ax.plot(sa.BASELINE["media_width2"], sa.BASELINE["k1"], "x", color="white", ms=7, mew=2.2)
    ax.plot(sa.BASELINE["media_width2"], sa.BASELINE["k1"], "x", color=INK, ms=6, mew=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"media attention width $\sigma^2_\mu$"); ax.set_ylabel(r"investment weight on expectations $k_1$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig_global")

    bins = [-1.0, 0.0, 0.2, 0.5, 1.0, 1.5]
    gb = pd.cut(base.gap, bins)
    over = base.groupby(gb, observed=True).any_overshoot.agg(["mean", "size"])
    wq = pd.qcut(base.media_width2, 4)
    out = dict(
        n_points=int(len(base)), n_runs=int(len(runs)), failed=int(runs.pattern.isna().sum()),
        pattern_share={sa.PATTERNS[k]: float(v) for k, v in base.pattern.value_counts(normalize=True).items()},
        overshoot_by_gap={str(k): dict(share=float(r["mean"]), n=int(r["size"])) for k, r in over.iterrows()},
        overshoot_when_gap_le_0=int(((base.gap <= 0) & (base.pattern >= 1)).sum()), n_gap_le_0=int((base.gap <= 0).sum()),
        cycle_share_among_overshoot=float((base.pattern == 2).sum() / (base.pattern >= 1).sum()),
        cycle_share_by_media_width_quartile=[float(v) for v in base.groupby(wq, observed=True).hype_cycle.mean()],
        ST=idx.pivot(index="factor", columns="outcome", values="ST").round(3).to_dict(),
        S1=idx.pivot(index="factor", columns="outcome", values="S1").round(3).to_dict(),
    )
    # boundaries around the baseline, read from the maps
    mw = pd.read_csv(RES / "sensitivity_map_w0_r_inf.csv")
    w_at = mw.w0.unique()[np.argmin(np.abs(mw.w0.unique() - sa.BASELINE["w0"]))]
    out["r_inf_max_with_cycle_at_baseline_w"] = float(mw[(mw.w0 == w_at) & (mw.pattern == 2)].r_inf.max())
    out["map_w_grid_point"] = float(w_at)
    ma = pd.read_csv(RES / "sensitivity_map_alpha2_r_inf.csv")
    r_at = ma.r_inf.unique()[np.argmin(np.abs(ma.r_inf.unique() - sa.BASELINE["r_inf"]))]
    sub = ma[ma.r_inf == r_at]
    out["alpha2_max_with_cycle_at_baseline_r"] = float(sub[sub.pattern == 2].alpha2.max()) if (sub.pattern == 2).any() else None
    out["alpha2_min_without_cycle_at_baseline_r"] = float(sub[sub.pattern != 2].alpha2.min()) if (sub.pattern != 2).any() else None
    for w in (1.0, 1.5, 2.0):
        wv = mw.w0.unique()[np.argmin(np.abs(mw.w0.unique() - w))]
        out[f"r_inf_max_with_overshoot_at_w_{w}"] = float(mw[(mw.w0 == wv) & (mw.pattern >= 1)].r_inf.max())
    numbers["global"] = out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    numbers = {}
    fig_noisy(numbers)
    fig_sensitivity(numbers)
    fig_dsmc_match(numbers)
    fig_fixed(numbers)
    numbers_size(numbers)
    fig_global(numbers)
    p = Params()
    numbers["params"] = {k: getattr(p, k) for k in ("a1", "a2", "b", "gam", "lam", "r_inf", "nu", "k1", "k2", "I0", "c")}
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
