# Metric Repair — session status & handoff

A consolidated snapshot of decisions, findings, and open theory that live in the *conversation* rather
than the code. Pairs with `OVERVIEW.md` (which documents the algorithms/formulations in detail). Drop
both into a Claude Project as knowledge, or read them to resume cold.

Repo: `AsafEtgar/metric-repair`. Everything below is committed and pushed. Focus is **non-complete graphs**
(G(n,p)); integer weights (geometric generator) unless noted.

---

## 1. What exists now (one-liner map; see OVERVIEW.md for detail)

- **Exact:** `domr_alg` (DOMR); `exact_metric_repair_ilp[_separation]` (GMR **and** IOMR via `iomr=`).
- **Fractional lower bound:** `metric_repair_lp_separation` (GMR/IOMR; `oracle="rsp"` exact / `"naive"` loose).
- **Approximations / heuristics:** `covering_lp_cover(solve∈{enum,separation} × rounding∈{deterministic,
  randomized,region_growing})`, `best_of_k` multi-vertex; `broken_cycle_rounding_heuristic`,
  `threshold_rounding_cover` (two corners of `covering_lp_cover`); `shortest_path_cover`; L1 family
  (`l1_min_heuristic`, `l1_rounding_heuristic`, and the new **cutting-plane** `l1_separation`); `pivot`,
  `left_edge`.
- **IOMR = hit each broken cycle at a LIGHT edge** (drop its unique max edge from the constraint). GMR =
  hit any edge. This one switch (`iomr=`/`drop_max=`) threads through every LP/ILP.

---

## 2. Guarantee status of every designed algorithm  ← the list to build proofs on

Legend: **✅ proven** (cite) · **📝 proven-in-principle** (standard argument, not yet written for us) ·
**❓ NEEDS PROOF** (we designed it; no ratio established) · **🔧 heuristic** (no ratio claimed/expected).

| Algorithm (variant) | Claimed guarantee | Status | What remains to prove |
|---|---|---|---|
| `domr_alg` | exact DOMR | ✅ | — (standard: edges with `w>dist`) |
| `exact_metric_repair_ilp(iomr=F/T)` | exact GMR / IOMR | ✅ | rests on Fan et al. Thm 6 (valid cover ⇔ hits every broken cycle, at a light edge for IOMR); we verified vs brute force |
| `exact_metric_repair_ilp_separation(iomr=F/T)` | exact on convergence | ✅ / 📝 | GMR: standard. IOMR: the integral oracle's **soundness+completeness** (don't-skip-cover-edges + drop-max = the `iomr_verifier` certificate) — we argued it; write it up cleanly |
| `metric_repair_lp_separation` (rsp) | `OPTfrac ≤ OPT`, exact LP optimum | ✅ | LP relaxation lower bound + RSP is exact separation ⇒ true optimum (paper Thm 4.1) |
| `covering_lp_cover(deterministic)` = `threshold_rounding_cover` | **`f`-approx**, `f=L` (GMR) / `L−1` (IOMR) `≤ W−1` | 📝 | The `f`-approximation of a bounded-frequency hitting-set LP: `|S| ≤ f·OPTfrac ≤ f·OPT` (round up `y_e≥1/f`). We proved it in-chat; it's textbook — **just needs writing** (and it *dominates the paper's `O(W log n)`*). Needs the exact (rsp) LP so `y*` is feasible for all cycles. |
| `covering_lp_cover(randomized)` = `broken_cycle_rounding_heuristic` | `O(log·OPT)` in expectation | ❓ | Textbook single-shot randomized rounding gives `O(log N_bc)=O(L log n)` **whp**. But our impl does **K union-rounds + a greedy oracle-driven top-up**, not one shot — the ratio/failure-probability of *that* procedure is **not established**. Prove a bound for the round-union+top-up variant (or switch to clean single-shot and cite). |
| `covering_lp_cover(region_growing)` | `O(log|H|)=O(log n)` under full separation | ❓ | Two gaps: (a) our impl is a **discretized** region-grow (level-sweep + min-boundary fallback) — prove it attains the continuous GVY bound (paper Thm 5.1/5.3 is for continuous radii); (b) **full separation empirically never holds** (§4), so any *unconditional* statement about our region-grow+top-up output is open. Likely low priority given (b). |
| `best_of_k` multi-vertex | keeps `f`-approx, often hits OPT | ✅(ratio) / 🔧(gain) | Ratio: each candidate is an optimal vertex ⇒ `f`-bound survives (trivial). The *empirical* OPT-recovery is a heuristic gain, no ratio. |
| `shortest_path_cover(general=T/F)` | "`L(+1)`-approx" (in code) | ❓ | The `L(+1)` claim is asserted in the docstring but **not verified for our implementation or for both variants** (there's a standing TODO: does `general=False`/IOMR behave as intended?). Confirm the proof (Fan et al. / Gilbert–Jain) applies to *this* code and each variant, or reprove. |
| `pivot_heuristic` (MVD pivot) | none | 🔧 / ❓ | The pivot has proven ratios for **correlation clustering / min-disagreement** (e.g. 3-approx). Whether that **ports to metric repair** is open — could be a real result. |
| `left_edge_heuristic` (Gilbert–Jain) | none | 🔧 / ❓ | Check whether Gilbert–Jain prove a ratio; if so, does our impl inherit it? |
| `l1_min_heuristic`, `l1_rounding_heuristic` | none | 🔧 | L1 is an L0 surrogate; L1/L0 has no general ratio. No proof expected. |
| **`l1_separation` (on G, our design)** | none (but provably **valid**) | 🔧 / ❓ | We proved the support is a **valid** cover on `G` (with `x≥0`, some light edge of every broken cycle is forced up; `w+x` metric). Empirically **much sparser** than completion-L1 and sometimes **= OPT** for IOMR. Open: is there *any* provable approximation ratio? (Probably not clean, but the sparsity vs completion-L1 might be characterizable.) |

### Structural / characterization open question (high value)
- **Is the GMR broken-cycle covering LP integral?** Empirically **zero integrality gap on every geometric
  instance** (any `n,p`), so `metric_repair_lp_separation` returns the *exact* GMR optimum there. But GMR
  metric repair is **NP-hard** (Fan–Raichel–Van Buskirk), so the LP **cannot be integral in general** —
  the observed integrality must be a property of the *instance class*. **Open: characterize when the
  broken-cycle covering polytope is integral, and prove geometric G(n,p) falls in that class** (or find the
  counterexample). If provable it would give poly-time exact GMR on that class. (Contrast: the **IOMR** LP
  has a genuine gap — Remark 5.4 / [Baier et al.] length-bounded-cut gap — so no such hope for IOMR.)

**The two ❓ rows I'd prioritize for a real theorem:** (1) the **randomized-with-top-up** ratio (tractable,
adapts standard set-cover rounding), and (2) the **GMR LP integrality** characterization (harder, higher
reward). The `f`-rounding proof (📝) is the easy win and is the one that **beats the paper's unconditional
bound**.

---

## 3. Per-algorithm 1-hour cluster reach (single core; cap is per (algorithm, instance))

**measured** = timed this session · **est.** = extrapolation.

| Algorithm | variant | ~n in 1 hr | basis |
|---|---|---|---|
| `domr_alg` | DOMR exact | 3000–5000 | one APSP (est.) |
| `metric_repair_lp_separation` (rsp) | **GMR exact** | ~1500–2000 @ p=0.2 · ~800–1000 @ p=0.3 | **measured** n=1000/p=0.2 = 11 min |
| `exact_metric_repair_ilp_separation` | GMR exact | ~1000+ | **measured** n=500/p=0.2 = 16 s |
| `exact_metric_repair_ilp_separation(iomr)` | **IOMR exact** | ~300–500 (may not converge dense) ⚠ | not timed at scale |
| `metric_repair_lp_separation(rsp, iomr)` | IOMR exact **LB only** | ~150 | **measured** n=100 = 100 s, n=200 DNF in 20 min |
| `metric_repair_lp_separation(naive, iomr)` | IOMR loose LB | ~2500–3000 @ p=0.5 | **measured** n=700/p=0.5 = 30 s |
| `covering_lp_cover` threshold, **rsp** (guaranteed `f`-approx) | IOMR | ~150 (rsp-LP-bound) | LP-limited |
| `covering_lp_cover` threshold, **naive** (no guarantee) | IOMR | ~2500 | tracks naive LP |
| `exact_metric_repair_ilp` / `broken_cycle_rounding` / `covering_lp_cover(enum)` | GMR/IOMR | ~100 @ p=0.2 · ~60 @ p=0.3 · ~50 dense | enumeration wall (1 hr barely helps) |
| `l1_separation` (on G) | GMR/IOMR | ~300–500 (est.) ⚠ | not timed at scale |
| `l1_min_heuristic` (completion) | IOMR/GMR | ~100 | memory-bound |
| `pivot` / `left_edge` (completion) | GMR/IOMR | ~300–500 (est.) | completion `O(n³)` |
| `shortest_path_cover` (on G) | GMR/IOMR | ~500–1000 (est.) | ~300 sparse baseline |

**Planning takeaways.** GMR exact scales to ~n=1000 comfortably. **Exact IOMR at ~1000 is out of reach with
a guarantee** — the rsp LP dies ~n=150 (gap ⇒ many rounds), so use either the separation **ILP** (no time
cap, week-partition) or the **naive-LP lower bound** (scales to ~3000 but loose). Everything
completion/enumeration-based is ≤ ~100–150 regardless of the hour. Density (p) shrinks all the
separation/enumeration numbers hard; the naive IOMR LP is the one measured *at* p=0.5. The two ⚠ rows
(exact IOMR ILP, `l1_separation`) are estimates — worth a timing probe before committing cluster hours.

---

## 4. The paper (§5 region-growing) review — verdict

The uploaded write-up ("Exact–Separation LP Approach to IOMR") is the **theory of exactly this codebase**:
its path-covering (LP) = our covering LP, its RSP separation = `_rsp_separation`, its `OPTfrac` =
`metric_repair_lp_separation`, its §7 GMR = our GMR variant. Proofs check out. Two substantive points:

1. **Its rounding bounds are dominated by trivial `f`-rounding (which we already implement).** The LP has
   row-sparsity `f = max|P| ≤ W−1`, so the textbook `f`-approximation gives `|S| ≤ (W−1)·OPTfrac = O(W)·OPT`
   **unconditionally** — a full `log n` better than the paper's `O(W log n)`. On the **WRG model** (`W=O(log n)`)
   that's `O(log n)` **unconditionally**, beating the paper's headline `O(log² n)`. Region growing only wins
   in the narrow regime **full separation holds AND `W` large**.

2. **Full separation — the paper's `O(log n)` precondition — empirically NEVER holds.** Swept the shortest
   `x*`-detour over every heavy pair: **`frac(detour ≥ 1) = 0`, median `= 0`** at every size/density (exact
   RSP `x*` for n≤100; naive `x*` for n=500–700, p=0.5–0.7, `|H|` up to ~43k). The LP concentrates mass on a
   sparse active set, so essentially every heavy pair has a near-zero-`x*` unbudgeted detour. So the
   region-growing route is unreachable here; `f`-rounding is the right IOMR approximation of record.

For the paper: either add `f`-rounding as the baseline and re-scope region growing to the large-`W`/
conditional case, or reconsider whether the WRG headline should be `O(log n)` via `f`-rounding.

---

## 5. Open items / next steps

**Theory (see §2 for the full guarantee list):**
- Write the `f`-rounding `O(W)` proof (easy; beats the paper). [📝]
- Prove a ratio for **randomized-rounding + top-up** as implemented. [❓, tractable]
- **GMR LP integrality** characterization on geometric G(n,p). [❓, high value]
- Confirm/port ratios for `shortest_path_cover` (L+1), `pivot`, `left_edge`. [❓]
- Primal–dual / region-growing scheme *with a usable ratio* (the paper's is vacuous here). [open]

**Empirical / engineering:**
- **Timing probe** for the two ⚠ rows (exact IOMR ILP, `l1_separation`) at p=0.3 and p=0.5, 1-hr cap, to
  firm up the cluster table.
- The **n≈1000 exact baseline** run (GMR is feasible; exact IOMR needs the ILP without a cap, or accept the
  naive LB) on the real ~1000-vertex datasets via `datasets.py`.
- `l1_separation` is sparser + validity-proven on G — consider making it the default L1 in experiments.
- Reweighted-L1 re-solve to sparsify (deferred).

**Resolved this session:** IOMR variants across all exact routines; the 2×2 `covering_lp_cover` grid +
multi-vertex; `l1_separation`; `l1(general=True)` now keeps `w+x` strictly positive (`min_weight=1`);
GVY region growing implemented; registry rounded out; full-separation sweep; dangling files cleaned.
