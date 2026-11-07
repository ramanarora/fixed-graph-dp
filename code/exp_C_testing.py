"""Experiment C: i.i.d. hypothesis testing on clustered networks.

The experiment measures error under the first Bernoulli hypothesis, with
fixed group size and varying numbers of groups. The exponential reference
curve isolates the privacy-noise term from the paper's finite-sample analysis.
Run with --plot-only to regenerate the figure from recorded JSON results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from multiattr import save_json
from figstyle import plt, COLORS, save

rng = np.random.default_rng(2028)
ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

n_i, C2 = 400, 0.24                      # fixed group size; beta = C2/sqrt(n_i)
beta = C2 / np.sqrt(n_i)
khs = [10, 25, 50, 100, 150, 220]
epss = [0.15, 0.3, 0.6]
T = 100_000

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plot-only", action="store_true",
                    help="plot recorded results without running Monte Carlo trials")
args = parser.parse_args()

if args.plot_only:
    with open(f"{RES}/exp_C.json") as f:
        rows = json.load(f)
else:
    rows = []
    for eps in epss:
        for kh in khs:
            n = n_i * kh
            # S_j = 1{#zeros in group j <= n/(2 kh)} under D_1 = Ber(1/2 - beta)
            zeros = rng.binomial(n_i, 0.5 + beta, size=(T, kh))
            S = (zeros <= n / (2 * kh))
            stat = S.sum(axis=1) + rng.laplace(0, 1 / eps, size=T)
            err = float(np.mean(stat > kh / 2))
            rows.append({"eps": eps, "kh": kh, "err": err, "T": T,
                         "x": eps * beta * np.sqrt(n * kh) / 4})
    save_json(f"{RES}/exp_C.json", rows)
fig, ax = plt.subplots(figsize=(4.1, 3.0))
floor = 1.0 / 100_000
xg = np.linspace(0, max(r["x"] for r in rows), 100)
ax.plot(xg, 0.5 * np.exp(-xg), "--", color="gray", lw=1.2,
        label=r"reference: $\frac{1}{2}e^{-\varepsilon\beta\sqrt{nk}/4}$")
for eps, mk, col in zip(epss, "os^", COLORS):
    xs = [r["x"] for r in rows if r["eps"] == eps]
    ys = [max(r["err"], floor) for r in rows if r["eps"] == eps]
    ax.plot(xs, ys, mk, color=col, label=rf"$\varepsilon={eps}$")
ax.set_yscale("log")
ax.set_xlabel(r"$\varepsilon\beta\sqrt{nk}/4$")
ax.set_ylabel("testing error (100k trials)")
ax.set_title("Group-based test: error collapses in " + r"$\varepsilon\beta\sqrt{nk}$")
ax.axhline(floor, color="gray", lw=0.6, alpha=0.5)
ax.text(0.1, floor * 1.3, "MC floor", fontsize=7, color="gray")
ax.legend()
save(fig, "expC_iid_testing", FIG)
