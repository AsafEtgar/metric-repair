# Metric Repair — implementation overview

A map of what is implemented: the graph models, the repair algorithms (by variant), the
LP/ILP formulations, and how each scales. Code lives in `graph_models.py`, `metric_repair.py`,
`metric_extras.py`; the experiment driver is `run_experiments.py`. See `README.md` for setup and
`equivalence/` for the Sage↔Python equivalence proof.

Terminology: a weighted graph is **metric** iff every edge weight equals the shortest-path distance
between its endpoints. A **broken cycle** is a cycle whose longest edge exceeds the sum of the rest
(`2·max > total`) — a violated polygon inequality. A **cover** `S` is a set of edges whose weights we
are allowed to change; `S` is *valid* iff some re-weighting of `S` makes the whole graph metric
(equivalently, `S` hits every broken cycle).

---

## 1. Graph models (`graph_models.py`)

| function | registry key | complete? | weights | metric? |
|---|---|---|---|---|
| `random_weighted_graph(n,p,1,100)` | `gnp_int` | no — G(n,p) | iid integer U[1,100] | **non-metric** |
| `random_geometric_weighted_graph(n,p)` | `geometric` | no — G(n,p) | iid Geometric(1−p), integer ≥ 1 | **non-metric** |
| `random_exponential_weighted_graph(n,p)` | `exponential` | no — G(n,p) | iid Exp(log 1/p), float ≥ 1 | **non-metric** |
| `random_uniform_weighted_graph(n,p)` | `uniform` | no — K_n thresholded (keep w>1−p) | float U(0,1) | **non-metric** |
| `random_metric_graph(n,p)` | — | no — connected G(n,p) | Euclidean dist. of random 5-d points, float | **metric** (control) |
| `uniform_complete_graph(n,L,U)` | — | **yes — K_n** | float U(L,U) | non-metric |
| `geometric_complete_graph(n,p)` | — | **yes — K_n** | Geometric(1−p), integer ≥ 1 | non-metric |
| `get_mst(G)` (transform) | — | no — tree | inherited | **metric** (tree) |

Notes
- **Metric generators** (`random_metric_graph`, `get_mst`) are controls: a Euclidean edge is always a
  shortest path, and a tree's unique paths are trivially shortest. Everything else has random weights
  and is non-metric (so it has broken cycles to repair).
- `geometric_complete_graph` was changed to draw Geometric(1−p) (support {1,2,…}); it no longer emits
  weight-0 edges (so the broken-cycle length bound applies — see §4).
- **You work mostly with non-complete graphs.** The general-graph algorithms (§2) run directly on
  `G`; only the *complete-only* cores (pivot, left-edge, L1) need `complete(G)` first.

---

## 2. Repair algorithms (`metric_repair.py`)

Three variants: **GMR** (general — increase or decrease), **IOMR** (increase-only),
**DOMR** (decrease-only). Feasibility predicates: `verifier(G,S)` (GMR) and `iomr_verifier(G,S)`
(IOMR, strictly stronger). Both are cheap (one shortest-path pass, no cycle enumeration).

| algorithm | variant | exact? | input | needs completion? | notes |
|---|---|---|---|---|---|
| `domr_alg(G)` | **DOMR** | **exact** | general | no | edges with `w > dist`; decreasing them is optimal DOMR. Also a valid GMR cover. |
| `exact_metric_repair_ilp(G)` | **GMR** | **exact** | general | no | min hitting set of broken cycles (ILP, §3). Exact ceiling ≈ n=100 sparse (§5). |
| `shortest_path_cover(G, general=True)` | **GMR** | heuristic | general | no | greedy L(+1)-approx; covers each broken edge + one shortest detour. |
| `shortest_path_cover(G, general=False)` | **IOMR** | heuristic | general | no | same, but covers only the detour edges (not the broken edge itself). |
| `pivot_heuristic(G)` → `MVD_Pivot(K)` | **GMR** | heuristic | **complete-only core** | yes | "min disagreement" pivot; seeded-reproducible. |
| `left_edge_heuristic(G)` → `Gilbert_Jain_IOMR(K)` | **IOMR** | heuristic | **complete-only core** | yes | Gilbert–Jain: fix the 'left' edge of each broken triangle. |
| `l1_min_heuristic(G, general=…)` → `l1_minimization(Kc)` | IOMR-oriented (default) or **GMR** (`general=True`) | heuristic | **complete-only core** | yes | L1 LP support (§3). |
| `l1_rounding_heuristic(G, general=…)` | **GMR** | heuristic | general (+completion for LP) | yes (LP step) | randomized rounding of the L1 LP. |
| `broken_cycle_rounding_heuristic(G)` | **GMR** | heuristic | general | no | randomized rounding of the broken-cycle covering LP (§3). |

**L1: two modes.** `general=False` (default) keeps the increase-oriented LP (`x ≥ 0` → weights only
increase; matches the Sage equivalence reference). `general=True` is true general MR: free-sign
corrections `x = x⁺ − x⁻`, minimise `Σ(x⁺+x⁻)`, weights may decrease (bounded so `w+x ≥ 0`). Rounding is
**sign-agnostic** — the cover is "which edges may change", so `l1_rounding_heuristic` samples on `|x_e|`
and the general `verifier` lets each chosen edge move either way; flipping `general` only swaps the LP.

`complete(G)` adds every missing edge `xy` with weight `dist_G(x,y)` (the metric completion; assumes
`G` connected). On a complete graph the only chordless cycles are triangles, so the L1 / left-edge /
pivot cores effectively work with triangle constraints.

---

## 3. Linear / integer programs

| name | type | constraints | variables | where | rounding |
|---|---|---|---|---|---|
| **L1 weight-correction** (`_l1_solve`, `l1_minimization`) | LP (`linprog`, `highs-ipm`) | polygon ineqs on `w+x` over chordless cycles of the completion (`induced_cycle_matrix`) | `x ≥ 0` (default) or free-sign `x=x⁺−x⁻` (`general=True`) | on `complete(G)` | support = cover; `l1_rounding_heuristic` |
| **Broken-cycle covering** (inside `broken_cycle_rounding_heuristic`) | LP relaxation (`linprog`, `highs`) | `B y ≥ 1`, `0 ≤ y ≤ 1` over broken cycles (`broken_cycle_incidence`) | `y` per edge | on `G` | randomized rounding + greedy top-up |
| **Exact hitting set** (`exact_metric_repair_ilp`) | ILP (`scipy.optimize.milp`) | `B y ≥ 1`, `y ∈ {0,1}` over broken cycles | `y` per edge | on `G` | — (exact) |

Constraint-matrix builders: `induced_cycle_matrix` (chordless-cycle polygon rows, for L1),
`broken_cycle_incidence` (broken-cycle hitting-set rows, for the ILP/covering LP),
`metric_triangles_matrix` (triangle rows, in `metric_extras.py`).

Solver note: the L1 optimum *value* is solver-independent; only *which* support is returned changes.
`highs-ipm` is marginally the sparsest single solve here; `reweight>0` (reweighted-L1) helps more.

---

## 4. The broken-cycle length bound (what makes the ILP tractable)

`broken_cycle_length_bound(G)` caps the length of any broken cycle: for positive weights a broken
cycle of length `k` satisfies `(k−1)·w_min < w_max`, so `k ≤ ⌊w_max/w_min⌋ + 1` (tightened to `w_max`
when weights are integers with `w_min=1`). This is a **weight** argument, so it holds on **any** graph,
complete or not. With `max_len=None` the enumeration auto-applies it and stays *complete* (loses no
broken cycle), so `exact_metric_repair_ilp` is genuinely exact. The bound caps cycle *length*, not
*count* — which is why it tames sparse graphs (few cycles) but not complete ones (≈ n^L cycles).

---

## 5. Scaling (single core; enumeration, not the solve, is the wall)

| algorithm | cost | practical ceiling |
|---|---|---|
| `domr_alg` | one APSP, `O(n³)` | thousands |
| `shortest_path_cover` on `G` | Floyd–Warshall × passes | n ≈ 300 (sparse) |
| `shortest_path_cover` on completion | `O(n³)`/pass on K_n (~`O(n^3.6)`) | n ≈ 150 (the suite bottleneck) |
| `pivot` / `left_edge` | completion + `O(n²–n³)` kernel | several hundred |
| `l1_min_heuristic` | completion + LP over `O(n³)` triangle rows | n ≈ 50–100 (memory-bound) |
| `exact_metric_repair_ilp`, `broken_cycle_rounding_heuristic` (on `G`) | broken-cycle enumeration | **n ≈ 100 @ p=0.2 · n ≈ 60 @ p=0.3 · ~50 denser** |

The exact ILP solve itself is always <0.1 s even at thousands of constraints; the wall is enumerating
broken cycles. That is exactly what the separation approach below removes.

---

## 6. Separation-based LP / ILP — V1 IMPLEMENTED (cutting planes)

**Goal.** Make exact / near-exact GMR scale past the enumeration ceiling (and onto larger non-complete
graphs) by never enumerating broken cycles up front. Solve the covering LP/ILP over a *growing* subset
of constraints, generated on demand.

**Implemented (metric_repair.py):**
- `metric_repair_lp_separation(G, oracle="rsp"|"naive")` → `(lp_value, y, D, n_cuts)`. Cutting-plane LP;
  `lp_value` is a valid **lower bound** on the exact cover size. `oracle="rsp"` (default) uses an EXACT
  weight-constrained-shortest-path separation (`_rsp_separation`, a weight-budget DP — pseudo-poly in
  w_max), so `lp_value` is the **true LP optimum over all broken cycles** (tightest bound); `"naive"`
  uses the canonical shortest-detour oracle (`_violated_cuts`, faster but loose).
- `exact_metric_repair_ilp_separation(G, max_rounds, time_limit)` → `(cover, info)`. Cutting-plane ILP
  with an **exact** verifier-based oracle: when it converges the cover is the **proven exact minimum**.
- `_violated_cuts`, `_rsp_separation` (oracles) + `_apsp_positions` / `_cuts_to_matrix` helpers.

Measured: the separation ILP matches the enumeration ILP exactly on small graphs and solves **n=200 to
proven optimality in 12–36 s** (enumeration died at n≈70–130), in 5–6 rounds. The RSP-LP equals the true
LP optimum on every test, and — crucially — **the hitting-set LP is INTEGRAL on these geometric
instances**: RSP-LP = exact ILP (178 at p=0.2, 535 at p=0.3; 0 % gap). So the **LP alone gives the exact
optimum** here, via a polynomial LP + pseudo-poly DP, sidestepping the NP-hard integer solve. (domr is
also near-optimal: 179 vs 178; 535 = 535.) The RSP DP is currently a pure-Python loop — the main
optimisation target for n=1000.

**Remaining / "go from there":** (a) **vectorise / batch the RSP DP** (the n=1000 bottleneck — group
directed edges by weight, drop the Python edge loop); (b) swap scipy `milp` for **Gurobi** (lazy
constraints) *if* the ILP is still needed where the LP isn't integral; (c) **restrict variables** to
edges on some cut; (d) rounding schemes (below) for an upper-bound cover when neither LP-integrality nor
the exact ILP closes.

**Cutting-plane loop.**
1. Start from a small constraint seed (e.g. broken triangles, or empty).
2. Solve the current LP relaxation `min 1·y s.t. (current rows) y ≥ 1, 0 ≤ y ≤ 1` (or the ILP).
3. **Separation oracle** — find a violated broken cycle, if any:
   - Interpret the LP solution `y` as edge "lengths"; for each edge `(u,v)`, look for a path from `u`
     to `v` whose `y`-length is `< y_{uv}` (or, for integrality/feasibility, use the `verifier`
     construction: heavy the chosen edges and find a non-cover edge undercut by a shortest detour).
     That edge + detour **is** a broken cycle violated by the current solution.
   - Add the most-violated cycle(s) as new rows; if none exist, the current solution is optimal/feasible
     over *all* broken cycles → **done, exactly**.
4. Repeat. Because only binding cycles are ever added, the constraint set stays small.

**Approximation / rounding schemes.** This must be *specific* — named schemes with stated guarantees,
not just "round the LP". Exact menu TBD with the user; candidate building blocks:
- Randomized rounding: sample edge `e` w.p. `min(1, scale·y_e)`, union over rounds, with the separation
  oracle as the validity check and a greedy top-up for any still-violated cycle (reuse the
  `broken_cycle_rounding_heuristic` machinery, but against the dynamic constraint set).
- Deterministic `f`-approximation for hitting set (round up every `y_e ≥ 1/f`, where `f` = max cycle
  length = the broken-cycle length bound `L`) — gives a provable `L`-factor cover.
- Primal–dual / region-growing on the cycle structure as an alternative with its own ratio.
- Threshold / iterated rounding as a deterministic baseline; reweighted-L1 re-solving to sparsify first.

⚠ **Spec the approximation guarantees explicitly before coding** — the user wants particular schemes
with particular ratios, not a generic rounding pass.

**Reuse.** The `verifier` already implements a separation oracle for *integral* covers (it finds an
undercut edge + detour = a violated broken cycle); generalising it to score a *fractional* `y` is the
main new piece. Constraint plumbing (`broken_cycle_incidence`, the `milp`/`linprog` calls) is in place.
