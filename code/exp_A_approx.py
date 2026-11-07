"""Contribution-bounding approximation on tight and planted-hub instances.

Panel (a) uses one adversarial order and twenty independent random orders.
Panel (b) compares the largest retained subset from eight runs of each method
on each frozen graph. The LP is solved once per graph and reused across runs.
"""
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np
import scipy
from multiattr import tight_instance, solve_lp, greedy_cb, lp_weighted_greedy, save_json
from figstyle import plt, COLORS, save

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)

rs = [2, 3, 4, 5, 6, 8, 10]
adv_ratio, rnd_ratio = {}, {}
for k in (1, 3):
    adv_ratio[k], rnd_ratio[k] = [], []
    for r in rs:
        n = 20 * r
        E, nV, order = tight_instance(r, n, k)
        opt = k * n
        adv_ratio[k].append(opt / len(greedy_cb(E, nV, k, order)))
        ratios = []
        for _ in range(20):
            order = rng.permutation(len(E)).tolist()
            ratios.append(opt / len(greedy_cb(E, nV, k, order)))
        rnd_ratio[k].append(float(np.mean(ratios)))

benchmark_path = ROOT / "data" / "contribution_bounding_instances.json"
benchmark = json.loads(benchmark_path.read_text())
repeats = 8
rows, timings = [], []
for i, instance in enumerate(benchmark["instances"]):
    E, nV, k = instance["edges"], instance["n_vertices"], instance["k"]
    started = time.perf_counter()
    y, opt = solve_lp(E, nV, k)
    lp_seconds = time.perf_counter() - started
    if not np.isclose(opt, instance["reference_lp_objective"], atol=1e-8):
        raise RuntimeError("LP objective disagrees with benchmark reference")
    greedy_rng = np.random.default_rng(np.random.SeedSequence([2029, i, 0]))
    weighted_rng = np.random.default_rng(np.random.SeedSequence([2029, i, 1]))
    counts = {"greedy": [], "lp_weighted": []}
    durations = {"greedy": [], "lp_weighted": []}
    for _ in range(repeats):
        started = time.perf_counter()
        selected = greedy_cb(E, nV, k, greedy_rng.permutation(len(E)).tolist())
        durations["greedy"].append(time.perf_counter() - started)
        counts["greedy"].append(len(selected))
        started = time.perf_counter()
        selected = lp_weighted_greedy(E, nV, k, y, weighted_rng)
        durations["lp_weighted"].append(time.perf_counter() - started)
        counts["lp_weighted"].append(len(selected))
        degrees = np.zeros(nV, dtype=int)
        for edge_id in selected:
            degrees[E[edge_id]] += 1
        assert degrees.max() <= k and len(selected) == len(set(selected))
    row = {key: instance[key] for key in ("planted_edges", "replicate", "maximum_degree", "hub_degree")}
    row.update({"lp_objective": opt,
                "fractional_lp_coordinates": int(np.count_nonzero((y > 1e-8) & (y < 1 - 1e-8))),
                "retained_counts": counts,
                "greedy_ratio": opt / max(counts["greedy"]),
                "lp_weighted_ratio": opt / max(counts["lp_weighted"]),
                "greedy_single_ratio": opt / counts["greedy"][0],
                "lp_weighted_single_ratio": opt / counts["lp_weighted"][0]})
    rows.append(row)
    timings.append({"instance": i, "lp_seconds": lp_seconds, "traversal_seconds": durations})

planted_counts = sorted({row["planted_edges"] for row in rows})
summary = []
for planted in planted_counts:
    group = [row for row in rows if row["planted_edges"] == planted]
    item = {"planted_edges": planted, "n_instances": len(group)}
    for method in ("greedy", "lp_weighted", "greedy_single", "lp_weighted_single"):
        values = [row[method + "_ratio"] for row in group]
        item[method] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    summary.append(item)

save_json(RES / "exp_A.json", {
    "rs": rs, "adv": adv_ratio, "rnd": rnd_ratio,
    "repeats_per_method": repeats, "algorithm_seed": 2029,
    "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
    "summary": summary, "instances": rows,
})
save_json(RES / "exp_A_runtime.json", {
    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    "timings": timings,
    "mean_lp_seconds": float(np.mean([row["lp_seconds"] for row in timings])),
    "mean_greedy_traversal_seconds": float(np.mean([v for row in timings for v in row["traversal_seconds"]["greedy"]])),
    "mean_weighted_traversal_seconds": float(np.mean([v for row in timings for v in row["traversal_seconds"]["lp_weighted"]])),
})

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax = axes[0]
ax.plot(rs, rs, "--", color="gray", lw=1.2, label=r"$y=r$ (worst case)")
ax.plot(rs, adv_ratio[1], "o-", color=COLORS[0], label=r"greedy, adversarial order ($k{=}1$)")
ax.plot(rs, adv_ratio[3], "s-", color=COLORS[3], ms=4, label=r"greedy, adversarial order ($k{=}3$)")
ax.plot(rs, rnd_ratio[1], "^-", color=COLORS[2], label=r"greedy, random order ($k{=}1$)")
ax.set_xlabel(r"maximum hyperedge size $r$")
ax.set_ylabel(r"approximation ratio $\mathrm{OPT}/|S|$")
ax.set_title("(a) Tight instances")
ax.legend()

ax = axes[1]
for method, fmt, color, label in (
        ("greedy", "o-", COLORS[0], "random-order greedy"),
        ("lp_weighted", "s-", COLORS[1], "LP-weighted greedy")):
    ax.errorbar(planted_counts,
                [row[method]["mean"] for row in summary],
                yerr=[row[method]["std"] for row in summary],
                fmt=fmt, color=color, label=label, capsize=2.5)
ax.axhline(1, color="gray", linestyle="--", lw=1)
ax.set_xscale("log")
ax.set_xticks(planted_counts, labels=[str(x) for x in planted_counts])
ax.set_xlabel(r"number of planted hub edges $q$ ($r=3$)")
ax.set_ylabel(r"$\mathrm{OPT}_{\mathrm{LP}}/|S|$")
ax.set_ylim(bottom=0.985)
ax.set_title("(b) Planted-hub instances (best of 8)")
ax.legend()
save(fig, "expA_r_vs_delta", FIG)
print(json.dumps(summary, indent=2))
