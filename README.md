# kinetic-hype

A kinetic (Boltzmann-type) model of innovation diffusion in which expectations, adoption and
technological performance co-evolve. Depending on parameters, expectations rise monotonically, overshoot
the realised value and settle, or overshoot, fall back and recover (a hype cycle). Two solvers share one
model definition:

* **Mean field** (`kinetic_hype.meanfield`): slow-time dynamics under time-scale separation, with a
  closed-form adoption hazard from the stationary Fokker–Planck law (an inverse gamma).
* **DSMC** (`kinetic_hype.dsmc`): direct simulation of N agents with the Nanbu–Babovsky scheme,
  vectorised in NumPy.

The code accompanies the working paper G. Masali, *Emergence of Hype-Disappointment Dynamics from
Social Interaction and Technological Feedback* (2026). It is not tied to any data set.

## Model

Agents hold a perceived value $v>0$ of an innovation. Per collision (coefficients $a_1,a_2,b,c\ge0$,
noise $\xi\sim N(0,\gamma b)$):

| agent | partner | update |
|---|---|---|
| non-user | non-user | $v' = v - a_1(v-v_*) - \mu(t)\,b\,(v-Pw(t)) + v\xi$ |
| non-user | user | $v' = v - a_2(v-v_*) - \mu(t)\,b\,(v-Pw(t)) + v\xi$ |
| user | anyone | $v' = v - c\,(v-Pr(t)) + v\zeta$ |

$\mu(t)$ is the media frequency, $w(t)$ the media message, $P(t)$ the price. A non-user with a purchase
opportunity (rate $\lambda$) adopts if $v>P$. The realised value is logistic in investment,

$$r=\frac{r_\infty}{1+e^{-\nu(I-I_0)}},\qquad I=k_1\!\int_0^t m_1\,ds+k_2\rho+I_{\rm exo}(t),$$

with $m_1$ the mean perceived value of non-users, $\rho$ the share of users and $I_{\rm exo}$ an exogenous
investment path.

**Mean field.** In the quasi-invariant limit, non-users' values have the stationary density
$f(v)\propto v^{-(2+\kappa)}e^{-\beta/v}$ with

$$m_1=P\frac{a_2\rho r+\mu b w}{a_2\rho+\mu b},\quad
\kappa=\tfrac{2}{\gamma}(\mu+\alpha_1\rho_1+\alpha_2\rho),\quad
\beta=\tfrac{2}{\gamma}(\alpha_1\rho_1m_1+\alpha_2\rho Pr+\mu Pw),$$

$\alpha_i=a_i/b$, $\rho_1=1-\rho$, and

$$\dot\rho=\lambda(1-\rho)\,\tilde\gamma(1+\kappa,\beta/P),\qquad \dot M=m_1 .$$

The media frequency $\mu(t)$ enters both $\kappa$ and $\beta$: once media attention fades, the
distribution is no longer pulled towards the message. Holding $\mu\equiv1$ in the hazard would overstate
adoption by about 17% at $t=20$ with the default parameters; `meanfield.adoption_hazard_unit_media` computes
that variant for comparison only.

## Install and run

```bash
pip install -e ".[dev]"
pytest                                   # 28 tests, about 40 s
python scripts/run_experiments.py        # DSMC results for notebooks 3-4, about 30 min on 2 cores
python scripts/make_paper_figures.py     # figures and numbers for the working paper (paper/figures/)
python scripts/run_sensitivity.py        # global sensitivity analysis for notebook 5, 15-20 min on 2 cores
jupyter lab notebooks/
```

```python
from kinetic_hype import Params, Signals, meanfield, dsmc, hype_features

p, s = Params(), Signals()                  # defaults: media bump at t=2, linear exogenous investment
mf = meanfield.simulate(p, s, T=20)         # DataFrame: t, rho, m1, r, investment, media, message, price
ab = dsmc.simulate(p, s, N=10_000, mu_c=4000, T=20, seed=0)
print(hype_features(mf))
```

Signals are plain functions of time; `gaussian`, `rise_then_flat`, `linear`, `constant` and
`fixed_signals` build common ones.

## Notebooks

1. `01_model` — the interaction rules, the aggregate system, the default hype cycle (paper Fig. 2), the
   distribution of perceived values at peak and trough, and a first DSMC run.
2. `02_hype_cycle` — the default hype cycle, why noise is needed for adoption, sensitivity to how fast
   media attention fades and to exogenous investment.
3. `03_dsmc_convergence` — DSMC against the mean field as relaxation speeds up and as N grows; the
   collision kernel's equilibrium against the inverse-gamma law.
4. `04_fixed_signals` — with constant media, the DSMC shows a hype cycle when relaxation is fast
   enough ($\mu_c\ge1000$ on the grid tested) but finite; the mean-field limit has no rising phase.
5. `05_sensitivity` — global (Sobol) sensitivity analysis of the aggregate system over 11 parameters: which
   factors decide whether expectations overshoot and whether the overshoot becomes a full cycle, and maps of
   the cycle region around the paper's baseline.

## Limitations

* **Time-scale separation is not satisfied at the default parameters with $\mu_c=1000$.** Expectations
  relax at rate $\mu_c(b\mu(t)+a_2\rho)$, which starts well below the purchase rate $\lambda$ and reaches only about
  $8\lambda$ late in the run. The mean field is the
  fast-relaxation limit, not a description of the $\mu_c=1000$ agents (notebook 3).
* **Under constant media the rising phase of the cycle depends on finite relaxation speed** and is absent
  in the mean field, which starts at the media equilibrium (notebook 4).
* The Fokker–Planck law is the small-jump limit of the collision kernel; large per-collision
  coefficients break it.
* $m_1$ is undefined when $\rho=0$ and $\mu=0$; media signals should not vanish exactly while nobody has
  adopted.
* Parameter values are illustrative and nothing is calibrated to data. The global sensitivity analysis
  (notebook 5) covers the aggregate system only, over ranges chosen by judgement. The default parameters lie inside the
  region where hype cycles occur, but a performance ceiling above about 0.9 removes the cycle.

## Background

* Pareschi, L. & Toscani, G. (2013). *Interacting Multiagent Systems: Kinetic Equations and Monte Carlo
  Methods*. Oxford University Press. (Kinetic opinion models, quasi-invariant limit, Nanbu and Babovsky
  DSMC schemes.)
* Toscani, G. (2006). Kinetic models of opinion formation. *Communications in Mathematical Sciences*
  4(3), 481–496.
* Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science* 15(5),
  215–227.

## License

MIT.
