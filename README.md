# Fixed-Graph Multi-Attribution Privacy

Code accompanying **Optimal Algorithms for Fixed-Graph Multi-Attribution Privacy** by Raman Arora and Kaibo Zhang, NeurIPS 2026. Please see the paper for the problem formulation, algorithms, and results.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{arora2026fixedgraph,
  author    = {Raman Arora and Kaibo Zhang},
  title     = {Optimal Algorithms for Fixed-Graph Multi-Attribution Privacy},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Getting started

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
python -m jupyterlab reproduce_figures.ipynb
```

The [notebook](reproduce_figures.ipynb) walks through figure generation step by step.

Run the implementation checks with:

```bash
PYTHONPATH=code python -m unittest discover -s tests -v
```

## Contribution bounding

`code/multiattr.py` represents a hypergraph as a list of attribution sets. Edge IDs
are list positions: two examples with the same attributed users are still two
different edges. Vertex IDs run from zero to `n_vertices - 1`, and an attribution
set lists each of its vertices once.

`lp_rounding_cb(E, n_vertices, k, rng, repeats=8)` solves the contribution-bounding
LP once. Each traversal orders positive LP weights by independent exponential
priorities and greedily accepts feasible edges, then considers zero-weight edges
in their original order. The largest output across the requested traversals is
returned. If every vertex already meets the cap, the full edge set is returned
directly. The cap `k` and the number of traversals `repeats` must be positive integers.

`lp_weighted_greedy` performs a single traversal when an LP vector is already
available. `greedy_cb` performs the feasibility checks in a supplied edge order.
All selectors use only the public attribution structure; the code does not
access private example contents during selection.

The LP is solved with SciPy's HiGHS backend. Numerically different optimal LP
solutions can induce different weighted orders, so a different solver or package
version can change retained sets even with the same random seed. The recorded
LP objectives provide a check on the optimization values. The implementation
clips returned coordinates to the interval `[0, 1]`, then enforces the integer
contribution cap explicitly while selecting edges.

## Contents

- `reproduce_figures.ipynb`: an executable walkthrough and figure regeneration.
- `code/multiattr.py`: LP, greedy selectors, graph generators, and Gaussian calibration.
- `code/exp_A_approx.py`, `code/exp_B_density.py`, `code/exp_C_testing.py`: experiment scripts.
- `code/figstyle.py`: shared plotting styles and figure export.
- `tests/test_contribution_bounding.py`: feasibility, maximality, exactness, edge-ID,
  boundary-case, and small-instance approximation checks.
- `data/`, `results/`, `figures/`: reproducibility inputs and recorded outputs.
- `requirements.txt`: pinned Python dependencies.
- `CITATION.cff`: citation metadata for the paper.
