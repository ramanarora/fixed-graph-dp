"""Experiment B (fixed-graph paper, Sec. 4): the network density parameter c
predicts the achievable mean-estimation error.

Across network families with matched n but c spanning three orders of
magnitude (star c=1, clustered c=1/#hubs, matching c=1/n), the two-phase
algorithm (full data, Gaussian mechanism at fixed-graph sensitivity
2*Delta/n = 2c) illustrates the Gaussian noise contribution
8*log(1.25/delta)*c^2*d/eps^2 in the mean-estimation upper bound.
"""
from pathlib import Path
import numpy as np
from multiattr import (star_hypergraph, clustered_hypergraph,
                       matching_hypergraph, gaussian_sigma, save_json)
from figstyle import plt, COLORS, save

rng = np.random.default_rng(2027)
ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

n, d, eps, delta = 1000, 10, 1.0, 1e-5
reps = 2000

families = [("star", 1.0, COLORS[3], "*")]
for kh in (2, 5, 10, 25, 50, 100, 250):
    families.append((f"clustered ({kh} hubs)", 1.0 / kh, COLORS[0], "o"))
families.append(("matching", 1.0 / n, COLORS[2], "s"))

cs, errs = [], []
for name, c, col, mk in families:
    Delta = int(round(c * n))          # max degree for these families
    sens = 2.0 * Delta / n             # fixed-graph l2-sensitivity of the mean
    sigma = gaussian_sigma(sens, eps, delta)
    X = rng.normal(0, 0.3, (n, d))
    X /= np.maximum(1, np.linalg.norm(X, axis=1, keepdims=True))
    noise = rng.normal(0, sigma, (reps, d))
    err = float(np.mean(np.sum(noise ** 2, axis=1)))
    cs.append(c)
    errs.append(err)

theory = [4 * c ** 2 * 2 * np.log(1.25 / delta) * d / eps ** 2 for c in cs]
save_json(f"{RES}/exp_B.json",
          {"families": [f[0] for f in families], "c": cs,
           "err": errs, "theory": theory})

fig, ax = plt.subplots(figsize=(4.1, 3.0))
cgrid = np.logspace(np.log10(1.0 / n), 0, 50)
ax.plot(cgrid, 4 * cgrid ** 2 * 2 * np.log(1.25 / delta) * d / eps ** 2,
        "--", color="gray", lw=1.2,
        label=r"theory: $8\ln(1.25/\delta)\,c^2 d/\varepsilon^2$")
for (name, c, col, mk), e in zip(families, errs):
    lbl = name if ("hubs)" not in name or "10 hubs" in name) else None
    if "hubs" in name and lbl is None:
        lbl = None
    ax.plot([c], [e], mk, color=col, label=lbl if lbl else None)
ax.plot([], [], "o", color=COLORS[0], label="clustered (varying hubs)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"network density parameter $c$")
ax.set_ylabel(r"$\mathbb{E}\|\hat\theta-\theta^*\|_2^2$")
ax.set_title(r"Error is governed by $c$, not $n$ ($n$ fixed)")
handles, labels = ax.get_legend_handles_labels()
seen, H, L = set(), [], []
for h, l in zip(handles, labels):
    if l and l not in seen:
        seen.add(l); H.append(h); L.append(l)
ax.legend(H, L, loc="upper left")
save(fig, "expB_density_parameter", FIG)
