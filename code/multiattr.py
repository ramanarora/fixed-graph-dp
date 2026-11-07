"""Contribution bounding and experiment helpers for fixed-graph privacy.

Hypergraphs are lists of edges, and edge IDs are list positions. Different
examples with identical attribution sets remain distinct edges. Randomized
selectors use explicitly supplied numpy.random.Generator instances.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

def random_hypergraph(n_vertices, n_edges, r_max, rng, r_min=1):
    """Random hypergraph with edge sizes uniform in [r_min, r_max]."""
    E = []
    for _ in range(n_edges):
        size = int(rng.integers(r_min, r_max + 1))
        E.append(sorted(rng.choice(n_vertices, size=size, replace=False).tolist()))
    return E


def star_hypergraph(n_edges):
    """All edges contain vertex 0 (center); c = 1."""
    return [[0, i + 1] for i in range(n_edges)], n_edges + 1


def clustered_hypergraph(n_edges, n_hubs, r=2):
    """n_hubs disjoint hubs; each cluster contributes n_edges/n_hubs edges,
    every edge contains its hub plus r-1 fresh leaves; c = 1/n_hubs."""
    per = n_edges // n_hubs
    E, vid = [], n_hubs
    for h in range(n_hubs):
        for _ in range(per):
            leaves = list(range(vid, vid + r - 1))
            vid += r - 1
            E.append(sorted([h] + leaves))
    return E, vid


def matching_hypergraph(n_edges, r=2):
    """Pairwise disjoint edges; c = 1/n."""
    E, vid = [], 0
    for _ in range(n_edges):
        E.append(list(range(vid, vid + r)))
        vid += r
    return E, vid


def tight_instance(r, n, k=1):
    """Grid construction from the fixed-graph paper: greedy with adversarial
    order achieves ratio exactly r (every hyperedge replicated k times).
    Returns (E, n_vertices, adversarial_order)."""
    assert n % r == 0
    vid = lambda j, i: j * r + i
    nV = n * r
    blocks = [[vid(l * r + q, 0) for q in range(r)] for l in range(n // r)]
    rows = [[vid(j, i) for i in range(r)] for j in range(n)]
    E = []
    for b in blocks:
        E += [b] * k
    for o in rows:
        E += [o] * k
    return E, nV, list(range(len(E)))


def incidence(E, n_vertices):
    A = np.zeros((n_vertices, len(E)))
    for j, e in enumerate(E):
        for v in e:
            A[v, j] = 1.0
    return A


def solve_lp(E, n_vertices, k):
    """Unregularized contribution bounding LP; returns (y*, OPT_LP)."""
    if not E:
        return np.empty(0), 0.0
    A = incidence(E, n_vertices)
    b = np.full(n_vertices, float(k)) if np.isscalar(k) else np.asarray(k, float)
    res = linprog(-np.ones(len(E)), A_ub=A, b_ub=b,
                  bounds=[(0, 1)] * len(E), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    y = np.clip(res.x, 0, 1)
    return y, float(y.sum())


def greedy_cb(E, n_vertices, k, order):
    """Greedy contribution bounding processing edges in `order`."""
    b = (np.full(n_vertices, k) if np.isscalar(k)
         else np.asarray([k.get(v, 10**9) for v in range(n_vertices)]))
    cnt = np.zeros(n_vertices, dtype=int)
    S = []
    for j in order:
        if all(cnt[v] < b[v] for v in E[j]):
            S.append(j)
            for v in E[j]:
                cnt[v] += 1
    return S


def lp_weighted_order(y, rng):
    """Size-biased permutation of positive LP weights, then zero weights.

    Independent unit exponentials divided by the positive weights give the
    exponential-race ordering. Zero-weight edges are appended in edge-ID order.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("LP weights must be a finite nonnegative vector")
    positive = np.flatnonzero(y > 0)
    # Log priorities avoid overflow when a solver returns a tiny positive weight.
    races = rng.exponential(size=len(positive))
    with np.errstate(divide="ignore"):
        keys = np.log(races) - np.log(y[positive])
    ordered = positive[np.argsort(keys, kind="stable")]
    return np.concatenate((ordered, np.flatnonzero(y == 0))).tolist()


def lp_weighted_greedy(E, n_vertices, k, y, rng):
    """One complete LP-weighted traversal, using a supplied feasible LP vector.

    Every edge is considered once. The output is feasible and maximal, even
    when the supplied LP vector has zero-weight coordinates.
    """
    if isinstance(k, (bool, np.bool_)) or not np.isscalar(k) or int(k) != k or k < 1:
        raise ValueError("k must be an integer at least one")
    if len(y) != len(E):
        raise ValueError("one LP weight is required per edge")
    return greedy_cb(E, n_vertices, int(k), lp_weighted_order(y, rng))


def lp_rounding_cb(E, n_vertices, k, rng, repeats=8):
    """Return the largest of independent complete LP-weighted greedy outputs.

    Solve the public-structure LP once and reuse its solution across traversals.
    This selection accesses no example contents. If the full edge set already
    meets the cap, return it directly without an LP solve or random sampling.
    """
    if isinstance(k, (bool, np.bool_)) or not np.isscalar(k) or int(k) != k or k < 1:
        raise ValueError("k must be an integer at least one")
    if (isinstance(repeats, (bool, np.bool_)) or not np.isscalar(repeats)
            or int(repeats) != repeats or repeats < 1):
        raise ValueError("repeats must be an integer at least one")
    if not E:
        return []
    degrees = np.zeros(n_vertices, dtype=int)
    for edge in E:
        degrees[edge] += 1
    if np.all(degrees <= k):
        return list(range(len(E)))
    y, _ = solve_lp(E, n_vertices, k)
    best = []
    for _ in range(int(repeats)):
        chosen = lp_weighted_greedy(E, n_vertices, int(k), y, rng)
        if len(chosen) > len(best):
            best = chosen
    return best


def gaussian_sigma(sensitivity, eps, delta):
    """Classical Gaussian mechanism calibration (valid for eps <= 1)."""
    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / eps


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


