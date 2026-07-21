# Roadmap

Living document. **Update the checkboxes and the "Current state" line as you go.**
Durable design facts live in [CLAUDE.md](CLAUDE.md); this file tracks progress only.

**Current state:** Day 0 planning and scaffolding complete. No implementation is currently in progress.

---

## How to use this file

- Work top to bottom. Do not start a later day before the current one's exit
  criteria are met.
- Each day is intended as one pull request, on the branch named in the heading.
  GitHub assigns PR numbers when a PR is opened; do not hardcode them here.
- If something is discovered that changes the design, update CLAUDE.md §6 (traps)
  in the same PR — that is how knowledge survives a context reset.
- If you want to do something not listed here, put it under "Future work". Do not
  start it. Scope creep is the main way a one-week project becomes a three-week
  project.

---

## Day 0 — Planning and scaffolding

- [x] `uv init`, Python 3.12, project skeleton
- [x] README.md with plain-language explanation, hypothesis, and non-claims
- [x] CLAUDE.md working context
- [x] ROADMAP.md (this file)
- [x] Dependencies pinned in `pyproject.toml`, `uv sync` clean
- [x] `.gitignore` covers `results/`

**Exit criteria:** `uv sync` succeeds from a clean clone.

---

## Day 1 — Generators, estimators, benchmark — branch `generators-and-harness`

### 1.1 Generators (`src/csid/generators.py`)
- [ ] Grouped equicorrelated generator (CLAUDE.md §3)
- [ ] AR(1) generator as **`G` independent stationary chains of length `m`**
- [ ] Linear embedding, and **graph-embedding** nonlinear map
      `phi(z) = R @ [z ; g(z)]` (CLAUDE.md §3) — full-rank Jacobian and injectivity
      hold by construction, so true ID is guaranteed to stay `d`
- [ ] `tests/test_embeddings.py`: Jacobian rank via SVD with a spectral gap, no local
      dimensional collapse, bi-Lipschitz sanity on the sampled region, and
      `omega -> 0` reproducing the linear case
- [ ] Gaussian copula wrapper for non-Gaussian marginals
- [ ] Emit **average pairwise within-block correlation** as a derived column on every
      run — `rho` is not comparable across the two arms, and this cannot be
      reconstructed after the fact

### 1.2 Property tests (`tests/test_generators.py`)
Tolerance-based, **not** KS tests (CLAUDE.md §6):
- [ ] One-point mean and covariance within tolerance of `N(0, I_d)` for every `rho`
- [ ] Selected quantiles within tolerance across `rho` — the one-point marginal must
      not move
- [ ] Empirical ICC matches target `rho` within tolerance
- [ ] **Joint structure does change as designed:** mean within-group squared distance
      tracks `2d(1 - rho)`. This is the positive control — if it fails, the dial isn't
      connected to anything.
- [ ] `m = 1` path is statistically indistinguishable from direct i.i.d. draws
- [ ] Each AR(1) chain is stationary from `t = 0` (no burn-in transient), and chains
      are mutually independent

### 1.3 Estimators (`src/csid/estimators.py`)
- [ ] `TwoNN_Regression` — implement **in-repo from the shared `r_1, r_2`**, not by
      calling `skdim` in the loop (CLAUDE.md §4a). The validation test must assert
      agreement with `skdim.id.TwoNN` under identical settings and truncation.
- [ ] `TwoNN_MLE` — implement directly, `d_hat = n / sum(log(mu))`
- [ ] `LevinaBickel` — record the averaging convention used
- [ ] `ParticipationRatio`
- [ ] The **three nearest-neighbour estimators** (`TwoNN_Regression`, `TwoNN_MLE`,
      `LevinaBickel`) share **one** kNN pass per dataset — the main compute saving.
      `ParticipationRatio` uses no neighbours at all and is computed separately from
      the covariance spectrum.
- [ ] Failure taxonomy defined and recorded for the three nearest-neighbour
      estimators: NaN, non-positive, `d_hat > D`, exception

### 1.4 Inference (`src/csid/inference.py`)
- [ ] Analytic interval for `TwoNN_MLE` (CLAUDE.md §4b), documented as exact **only
      under the idealized i.i.d.-Pareto model for the ratios**
- [ ] **i.i.d.-Pareto coverage sanity check.** Simulate `mu_i` directly as i.i.d.
      Pareto(1, d) — bypassing all geometry — and confirm empirical coverage hits
      nominal. This validates the interval machinery on its own terms and isolates
      implementation bugs from both the shared-neighbour effect and clustering.
- [ ] **Validate numerically against R `intRinsic`**, with **trimming and
      bias-correction settings explicitly matched on both sides**. `intRinsic` trims by
      default; note whether `n/S` or the unbiased `(n-1)/S` convention is in use, since
      `E[n/S] = n*d/(n-1)`. Record the comparison in the PR description. Correctness
      gate, not a nicety.

### 1.5 Jackknife machinery (`src/csid/jackknife.py`)
To be built here rather than on Day 4, because §1.6 will benchmark it and you cannot
benchmark what does not exist yet. Day 4 will only *run* the experiment.
- [ ] Leave-one-group-out interval per CLAUDE.md §5: centre `theta_hat`,
      `SE = sqrt(Var_jack)`, **both `t_{G-1}` and `z` critical values** emitted
- [ ] Cached-neighbour-list fast path with `K = m + k_max`, `k_max` recorded per run.
      Day 4 will run `TwoNN_MLE` only, so `k_max = 2` and `K = m + 2` there.
- [ ] Fallback triggers when any retained point has `< k_max` surviving neighbours;
      implement it as an **assertion**, since equal group sizes make it unreachable
- [ ] Whole-trial failure semantics — any failed replicate fails the trial
- [ ] Optional log-scale variant (same replicates, different transform)
- [ ] **Equivalence test** (`tests/test_jackknife.py`): cached path numerically
      identical to naive full recomputation on a small case. Correctness gate.

### 1.6 Benchmark before committing to any grid
- [ ] Time one representative Stage A condition end to end
- [ ] Time **one complete leave-one-group-out jackknife trial** — `G = n/m` estimator
      recomputations — at both `m = 10` (`G = 150`) and `m = 50` (`G = 30`), since cost
      per trial scales with `G`
- [ ] Record the cached-vs-naive speedup alongside the equivalence result
- [ ] Write measured throughput into "Measured runtimes" below
- [ ] Recompute Day 2 and Day 4 grid sizes from **measured** numbers and revise them
      here before running anything

### 1.7 Pilot grid → engineering and signal check (NOT a kill gate)
Verifies correctness, runtime, effect direction, and whether magnitudes justify the
planned Stage A grid. **A formal null conclusion can only come from Stage A** — see
S1/S1b. Small grid, all three signals, **not** bias alone:
- [ ] `d` in {5, 10, 20}, `rho` in {0, 0.6, 0.9}, `m` in {1, 10, 50}, grouped, linear
      embedding, `n = 2000` fixed, 60 seeds
- [ ] For the three nearest-neighbour estimators, report bias difference, variance
      ratio, RMSE, and failure rate. For participation ratio in this isotropic linear
      Gaussian pilot, report bias and RMSE against true `d`.
- [ ] For `TwoNN_MLE` only, report analytic-interval coverage **both** absolutely and
      as change relative to the `rho = 0` control.
- [ ] Record the decision and its justification in "Decisions log" below

**Exit criteria:** tests pass, intervals validated against `intRinsic`, measured
runtimes recorded, pilot decision logged.

---

## Day 2 — Stage A factorial — branch `stage-a-bias-variance`

- [ ] Full grid (sizes to be confirmed by the Day 1 benchmark):
      `d` in {2, 5, 10, 20} x `rho` in {0, 0.3, 0.6, 0.9} x `m` in {1, 2, 5, 10, 25, 50}
      x {grouped, AR1} x {linear, nonlinear} x **150 seeds**, `n = 2500` fixed
- [ ] Fixed-`G` slice as a separate run (never merged into a fixed-`n` figure)
- [ ] `TwoNN_MLE` analytic-interval coverage computed for every condition — it is
      closed-form and therefore nearly free, so the full coverage surface comes out
      of this day
- [ ] Raw per-run parquet written to `results/raw/` (gitignored)
- [ ] Compact per-condition summary written to `results/summaries/` (**committed**)

### Preregistered decision thresholds

These thresholds are written before Stage A results are observed and must not be
revised after seeing them. The primary decision panel is **grouped dependence with a
linear Gaussian embedding**, with each dependent condition compared with its matched
`rho = 0` control. AR(1), nonlinear embeddings, and transformer results are secondary
for the go/no-go decision. Coverage means `TwoNN_MLE` coverage; bias and variance are
evaluated separately for the three nearest-neighbour estimators. Participation ratio
is reported under its embedding-specific scoring but is outside the three-signal null
gate.

- **Coverage:** a decrease of at least **5 percentage points**, with a 95% Monte Carlo
  confidence interval for the difference excluding zero.
- **Bias:** `abs(delta_bias) / d >= 0.05`, with a 95% Monte Carlo confidence interval
  for the change excluding zero.
- **Variance:** a dependent/control ratio of at least **1.25** or at most **0.80**,
  with a 95% Monte Carlo confidence interval excluding 1.
- **Coherence:** at least one meaningful effect must form a pattern across two or more
  neighbouring dependence conditions, such as increasing `rho` or `m`; an isolated
  grid cell cannot decide the gate.
- **Saturated controls:** label matched `rho = 0` coverage at or below **20%** as
  floor-limited and at or above **99%** as ceiling-limited. Report these contrasts,
  but do not use either as the sole go/no-go evidence. If the matched control has
  `abs(bias) / d >= 0.20`, label it substantially control-biased and interpret its
  coverage as an estimator-baseline limitation.

Stage A supports a null-result stop only if coverage, bias, and variance all fail to
meet their thresholds as coherent patterns in the primary panel. Always report
failure rates; they are outside the three-signal null gate unless failures prevent
meaningful evaluation.

**Exit criteria:** raw parquet exists locally, committed summary reproduces every
applicable outcome per cell without touching `results/raw/`.

---

## Day 3 — Implied effective sample size — branch `neff-estimation`

- [ ] For the three nearest-neighbour estimators, build i.i.d. reference curves for
      bias and variance vs. `n` at `rho = 0`
- [ ] Invert each applicable dependent condition onto the reference curve → implied
      `n_eff`
- [ ] Do this **separately from bias and from variance**. If the two disagree, that is
      a finding about which moment degrades first, not an error to average away.
- [ ] Overlay Kish `n_eff = n / (1 + (m-1) * rho)` — **linear Gaussian grouped panel
      only** (CLAUDE.md §6)

**Checkpoint (S2):** if implied `n_eff` is non-monotonic in `m` or unstable across
seeds, the inversion is ill-posed. Fall back to reporting raw bias/variance surfaces
and drop the `n_eff` framing. Log the decision.

---

## Day 4 — Cluster-aware intervals — branch `jackknife-intervals`

Scope note: **group subsampling was cut before implementation** and moved to Future
work. Subsampling inference needs the estimator's convergence rate to rescale
subsample estimates, and under grouped sampling that rate is the effective sample size
this project exists to measure — so any scaling rule would assume the answer. Jackknife
is the primary cluster-aware method.

The jackknife machinery, its fast path, and its equivalence test are to be built and
benchmarked on Day 1 (§1.5–1.6). **This day will only run the coverage experiment.**

- [ ] Reduced grid: `d` in {5, 10} x `rho` in {0, 0.6, 0.9} x `m` in {10, 50},
      grouped only, `n = 1500`, `T = 400` trials per condition (roughly +/-2.5%
      standard error on a coverage estimate). `m = 1` is excluded here: with `G = n`
      groups, leave-one-group-out degenerates to leave-one-point-out and costs `n`
      recomputations per trial for no added insight.
- [ ] Compare analytic vs. jackknife, reporting **both absolute coverage and change
      relative to each method's own `rho = 0` control** — absolute says whether the
      interval is usable at all, relative carries the causal attribution to clustering
- [ ] Report jackknife coverage under **both `t_{G-1}` and `z`** critical values. Free
      from identical replicates, and settles empirically which is better calibrated
      rather than assuming it.
- [ ] Log-scale variant coverage — **first thing to drop if the day overruns**

**Jackknife is being tested, not assumed.** It is standard for clustered survey data,
but the jackknife is known to fail for non-smooth statistics and nearest-neighbour
distances are non-smooth — a point's nearest neighbour changes discontinuously as the
sample changes. If jackknife coverage is also poor, report it. "Neither interval
survives clustering" is a legitimate and useful finding; do not go hunting for a third
method to rescue the narrative.

**Descope trigger (S3):** if incomplete at end of Day 4, ship analytic coverage only
and move jackknife to Future work. Do not let this arm eat the writeup.

---

## Day 5 — Transformer protocol sensitivity — branch `transformer-protocols`

Controls, all required:
- [ ] **One fixed parent corpus**, one **fixed model checkpoint**, identical
      preprocessing and token-eligibility rules across every protocol
- [ ] **Total activation count held fixed**; vary sequences x tokens-per-sequence:
      (4000,1), (400,10), (80,50), (16,250)
- [ ] Protocols: one-token-per-sequence, multiple-tokens, sequence-stratified,
      fixed-position, token-random
- [ ] **Repeated protocol resampling** — many draws per protocol. Report the
      **distribution** of estimates per protocol, not a single point estimate per
      protocol.

**Required limitation statement, to appear in the report and in any figure caption:**

> This stage measures **practical protocol sensitivity** and **cannot causally isolate
> dependence**. Changing the number of sequences also changes sequence and token
> composition — 16 sequences sample far less corpus diversity than 4,000 — so
> dependence and composition move together by construction. No claim about the true
> intrinsic dimension of the representation is made.

**Hard timebox: one day.** If activation extraction is slow, shrink the model or cut
the arm entirely. This is a demonstration, not the contribution. No claim about true
ID (CLAUDE.md §6).

---

## Day 6 — Report — branch `report`

- [ ] `docs/REPORT.md`: question, design, results, limitations, what would falsify it
- [ ] Figures regenerated from the **committed summary files** in `results/summaries/`
      with no access to `results/raw/`, git SHA recorded on each
- [ ] Prior-work positioning with gap boundaries drawn explicitly
- [ ] Limitations stated up front, not buried
- [ ] README status updated, headline result summarised in plain language

---

## Standing stopping criteria

| ID | Trigger | Action |
|---|---|---|
| **S1** | Day 1 pilot — **engineering and signal gate, not a kill gate** | Check four things: (a) correctness — invariants and validation tests pass; (b) runtime within budget; (c) effect *direction* consistent with hypothesis; (d) effect *magnitude* large enough that the planned Stage A grid is adequately powered. If magnitude looks marginal, **re-scope the grid** (more seeds, stronger `rho`/`m`) before Stage A. **Do not conclude a null here** — 60 seeds gives a variance-ratio standard error near 26%, which cannot support one. |
| **S1b** | In the primary Stage A panel, coverage, bias, and variance all fail their preregistered thresholds as coherent patterns | Stop. Write it up as a null result; the i.i.d. reference-curve calibration remains a usable artifact. **This is the only kill gate.** |
| **S2** | `n_eff` inversion ill-posed | Report raw surfaces, drop `n_eff` framing |
| **S3** | Day 4 resampling arm overruns | Analytic coverage only; resampling to Future work |
| **S4** | Any new experiment idea | Write it under Future work. Do not start it. |
| **S5** | Any arm exceeds one overnight run | Cut the **grid**, never the seed count |

S1b requires **all three** signals to miss their thresholds under the coherence rule.
Any one qualifying coverage, bias, or variance pattern is sufficient reason to
continue. Coverage is the primary outcome, and clustering can destroy coverage while
leaving bias almost untouched.

---

## Measured runtimes

Filled in by Day 1 step 1.5. **Do not lock any grid until these exist.**

| Operation | Config | Measured | Notes |
|---|---|---|---|
| Single Stage A dataset | _tbd_ | _tbd_ | generate + one kNN pass + 3 NN estimators + PR |
| One jackknife trial, `m=10` | `n=1500`, `G=150` | _tbd_ | `G` recomputations, cached neighbour lists |
| One jackknife trial, `m=50` | `n=1500`, `G=30` | _tbd_ | `G` recomputations, cached neighbour lists |
| Naive vs. cached speedup | small case | _tbd_ | must also be numerically identical |
| Projected Stage A total | _tbd_ | _tbd_ | |
| Projected Day 4 total | _tbd_ | _tbd_ | 12 conditions x `T=400` trials x `G` |

---

## Design freeze

**The current written experimental plan is frozen during implementation.**

During implementation, **do not propose design changes**. Any improvement goes to
Future work and does not block the current plan.

The freeze reopens **only** for these five triggers:

1. A mathematical invariant or property test fails.
2. Validation against `intRinsic` or `scikit-dimension` fails.
3. Cached and naive jackknife implementations disagree.
4. Measured runtime makes the planned grid infeasible.
5. Directly duplicative prior work is discovered.

If one fires: stop, record it in the decisions log with evidence, make the minimum
change that resolves it, and resume. Nothing else is grounds for reopening.

---

## Decisions log

Append-only. One line each: date, decision, reason.

- **2026-07-19** — Primary outcome is interval coverage, not bias. Clustering removes
  information without removing rows, so overconfidence can appear with negligible bias.
- **2026-07-19** — Two TwoNN variants carried separately after verifying
  `skdim.id.TwoNN` is the Facco regression estimator with 10% truncation, not the MLE.
- **2026-07-19** — Bootstrap with replacement excluded outright; duplicate vectors give
  `r_1 = 0` and break every NN-based estimator.
- **2026-07-19** — Kish's design effect demoted to a reference line on the linear
  Gaussian grouped panel only.
- **2026-07-19** — Analytic interval documented as exact only under the idealized
  i.i.d.-Pareto model for the ratios, not merely under independent sampling of points.
  Ratios are dependent even for i.i.d. points because neighbours are shared. Coverage
  is therefore reported as degradation **relative to the `rho = 0` control**; an
  i.i.d.-Pareto sanity check validates the interval machinery separately.
- **2026-07-19** — Framing corrected throughout: the invariant is the **one-point
  marginal**, not "the same data." Joint structure and pairwise geometry change by
  design and are the mechanism under study. Added a positive control test asserting
  mean within-group squared distance tracks `2d(1 - rho)`.
- **2026-07-19** — Dropped the claim that participation ratio exceeds `d` on curved
  manifolds; it can fall either side, since unequal eigenvalues push PR below `d` while
  curvature pushes it up. Scored against its own baseline as already planned.
- **2026-07-19** — AR(1) arm defined as `G` independent stationary chains of length
  `m`, matching the grouped block structure. `rho` is not comparable across arms;
  average within-block pairwise correlation recorded as a derived column.
- **2026-07-19** — **Group subsampling cut before implementation.** Subsampling
  inference requires the estimator's convergence rate to rescale subsample estimates;
  under grouped sampling that rate is the effective sample size the project exists to
  measure, so any scaling rule assumes the answer. Leave-one-group-out jackknife is
  the sole cluster-aware method. Moved to Future work.
- **2026-07-19** — Jackknife documented as **under test, not assumed valid**. It is
  standard for clustered survey data but is known to fail for non-smooth statistics,
  and nearest-neighbour distances are non-smooth. "Neither interval survives
  clustering" is an acceptable finding; do not hunt for a third method to rescue a
  narrative.
- **2026-07-19** — Result storage policy fixed: `results/raw/` gitignored,
  `results/summaries/` committed, every figure reproducible from committed summaries
  alone. Resolves a contradiction between the Day 2 and Day 6 entries.
- **2026-07-19** — Novelty phrasing standardised to "we found no prior work that
  directly measures X" rather than "nobody has measured X". The second is
  unverifiable and one citation away from being false.
- **2026-07-19** — ~~Jackknife **interval** fully specified, not just its variance:
  centre on the full-sample estimate, `SE = sqrt(Var_jack)`, critical value
  `t_{G-1}` rather than normal `z`, on the grounds that `z` would under-cover and
  contaminate the measurement.~~ **SUPERSEDED** — the interval spec stands, but the
  claim that `z` is wrong was an overclaim; see the later entry. Coverage is
  undefined without a full interval specification, which was the durable part.
- **2026-07-19** — ~~Added a cached-neighbour-list method for the jackknife, top
  `K = m + 20` retained.~~ **SUPERSEDED** on the cache depth — see the later
  `K = m + k_max` entry. The durable part: `O(n*K)` per replicate instead of
  `O(n^2)`, roughly two orders of magnitude faster, to be verified numerically
  identical to naive recomputation. Without it the Day 4 grid is ~432,000 estimator
  fits and would trigger S3 for purely implementation reasons.
- **2026-07-19** — Replicate failures fail the **whole trial**. Computing variance over
  surviving replicates only would bias it downward, which is indistinguishable from
  the overconfidence the project is trying to detect.
- **2026-07-19** — Coverage reported **both** absolutely and relative to the `rho = 0`
  control. Reverses an earlier over-correction that said relative only: relative alone
  hides whether the interval is usable at all, absolute alone misattributes the known
  shared-neighbour effect to us.
- **2026-07-19** — Cache depth changed from the arbitrary `K = m + 20` to
  `K = m + k_max`, where `k_max` is the largest neighbourhood any estimator in the run
  needs. A removed group deletes at most `m` points, so `k_max` neighbours are
  guaranteed to survive — the buffer is tight and principled. Fallback now triggers on
  `< k_max` survivors rather than total deletion, and is **provably unreachable** under
  equal group sizes, so it will be implemented as a bug assertion. `k_max` must be
  recorded per run, since changing Levina–Bickel's `k` silently changes the required
  depth.
- **2026-07-19** — Nonlinear embedding specified as a **graph embedding**
  `phi(z) = R @ [z ; g(z)]`. The identity block forces a rank-`d` Jacobian for every
  `g` and every `z`, and the first `d` coordinates recover `z`, so injectivity and full
  rank hold **by construction** rather than by inspection. True ID is guaranteed to
  remain `d`; the numerical checks become regression guards. Curvature knob is
  `omega`, with `omega -> 0` recovering the linear case.
- **2026-07-19** — `t_{G-1}` language softened. §5 had asserted `t_{G-1}` was correct
  and that `z` would "contaminate" results, contradicting the same section's statement
  that the jackknife is under test. Neither critical value is theoretically guaranteed
  for a nonlinear, non-smooth estimator under clustering. `t_{G-1}` is now the
  pre-registered candidate, and **both `t` and `z` coverage are reported** — free from
  identical replicates, so the choice is settled by measurement.
- **2026-07-19** — Public docs said "three estimators" while the design carries four.
  Corrected to four (three families, two distinct TwoNN variants) with a table naming
  each.
- **2026-07-19** — Jackknife implementation and equivalence test moved from Day 4 to
  Day 1 §1.5. Day 1 benchmarks it, so it cannot be written on Day 4. Day 4 now runs
  only the coverage experiment.

---

- **2026-07-19** — Outcome scope pinned: bias, variance, RMSE, and failure rate for
  the three nearest-neighbour estimators; participation-ratio bias/RMSE against true
  `d` only for the isotropic linear Gaussian embedding, and matched-baseline change on
  nonlinear embeddings; analytic **and** jackknife interval coverage for `TwoNN_MLE`
  only. Jackknife is generic and could wrap any estimator — restricting it is a scope
  decision that quarters Day 4 cost and keeps analytic-vs-jackknife like-for-like.
  Implies `k_max = 2`, `K = m + 2` for the Day 4 run.
- **2026-07-19** — `TwoNN_Regression` selected for in-repo implementation from the
  shared `r_1, r_2` rather than calling `skdim.id.TwoNN` in the loop. skdim computes
  its own distances internally, so wrapping it would contradict the shared-kNN design.
  The planned validation test will assert agreement under matched settings; note
  `Femp` divides by `N`, not the truncated count.
- **2026-07-19** — Day 1 pilot demoted from kill gate to **engineering and signal
  gate**. At 60 seeds a variance-ratio estimate carries ~26% standard error, which
  cannot support a null. Added **S1b** at Stage A (150 seeds) as the sole kill gate;
  without it, removing S1 would have left no kill criterion at all.
- **2026-07-19** — Embedding parameters frozen within every comparison slice: same
  `R`, `b_j`, amplitudes, phases, and `omega` across all `rho` and `m` being compared,
  drawn from a dedicated `embedding_seed` recorded on every run row. Redrawing per
  condition would vary the manifold with the treatment and confound everything. The
  current plan uses one realization per slice; multi-realization robustness is Future
  work.
- **2026-07-19** — Transformer stage controlled: one fixed corpus, one fixed
  checkpoint, identical preprocessing and eligibility rules, repeated protocol
  resampling reporting distributions rather than point estimates. Required limitation
  statement added — the stage cannot causally isolate dependence, because changing the
  number of sequences also changes sequence and token composition.
- **2026-07-19** — **DESIGN FREEZE.** The written plan was frozen before
  implementation. It reopens only on: invariant/property-test failure, `intRinsic` or
  `scikit-dimension` validation failure, cached-vs-naive jackknife disagreement,
  measured runtime making the grid infeasible, or discovery of directly duplicative
  prior work. All other improvements go to Future work.
- **2026-07-21** — Stage A practical-effect thresholds, Monte Carlo confidence
  criteria, primary decision panel, coherence rule, and saturated-control handling
  preregistered before implementation. They must not be revised after Stage A results
  are observed.

## Future work

Parking lot. Nothing here gets started during the one-week build.

- Unions of manifolds and mixed intrinsic dimensions
- Near-duplicate contamination as a separate dependence mechanism
- Whether the two TwoNN variants respond differently to clustering (if the Day 2
  results hint at it, this becomes a natural follow-up)
- Evaluate L2N2 (Ong et al., 2026) under clustered sampling; it is adjacent recent work
  and is not part of the current experimental grid
- Non-Gaussian marginals beyond the copula smoke test
- Larger models, GPU scale
- **Group subsampling intervals.** Cut before implementation: subsampling requires the
  estimator's convergence rate, which under grouped sampling is the effective sample
  size we are trying to measure — so any scaling rule assumes the answer. A
  non-circular version would estimate the rate empirically across several subsample
  sizes (Politis–Romano–Wolf), which is its own project.
