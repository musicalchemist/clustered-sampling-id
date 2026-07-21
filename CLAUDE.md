# Working context for agents and contributors

@AGENTS.md

For Claude Code, the import above supplies the repository-wide workspace, workflow,
and security rules. Other tools should read [AGENTS.md](AGENTS.md) directly.

Read this file **and** [ROADMAP.md](ROADMAP.md) before making changes. This file holds
the durable facts: what is true, what is decided, and what has already burned us.
ROADMAP.md holds the changing state: what is done and what is next.

If you are an AI agent picking this up cold, everything you need is in these two files
plus the docstrings. Do not re-derive the design from scratch — it has already been
through several rounds of review, and the non-obvious choices below are deliberate.

---

## 1. What this project is

The plan defines an empirical study of how **clustered (grouped) sampling** affects
intrinsic-dimension estimators. Bias, variance, RMSE, and failure rate apply to the three
nearest-neighbour estimators; participation ratio is an embedding-specific contrast;
and the primary outcome, **confidence-interval coverage**, applies to `TwoNN_MLE`.

Motivating case: people estimate the ID of neural network representations by sampling
many token activations from a small number of sequences. Tokens within a sequence are
dependent. The estimators assume independence. We found no prior work that directly
measures this exact finite-sample estimator and interval behaviour under externally
imposed clustered sampling.

**Phrase novelty this way, always.** We searched; we did not find it. That is not the
same as proving it does not exist, and absolute claims ("nobody has measured this")
are both unverifiable and easy to falsify with one citation. Cautious phrasing costs
nothing and cannot be embarrassed.

**Primary outcome is coverage, not bias.** Clustering removes information without
removing rows, so an interval that counts rows will be overconfident even when the
point estimate is fine. Do not let the project drift into a bias-only study.

---

## 2. Non-negotiable invariants

These are the things that, if broken, silently invalidate every result. Guard them
with tests, not vigilance.

1. **The one-point marginal must be identical across all dependence conditions;
   the joint distribution deliberately is not.**
   Every comparison is "same one-point marginal, different joint dependence
   structure." Do **not** describe this as "the same data with different grouping" —
   that is false and it hides the mechanism. Pairwise geometry genuinely changes: for
   two points in the same group, `E||z_1 - z_2||^2 = 2d(1 - rho)`, so groupmates move
   closer as `rho` grows, become likelier to be each other's nearest neighbours, and
   do so at shorter distances. That is the causal pathway under study, not a confound.
   What must not change is the distribution of a single point examined on its own.
   Guaranteed by *construction* (§3), checked by tolerance tests.
2. **`m = 1` is the i.i.d. control**, produced by the same code path as every other
   condition. Never add a separate "independent" generator — a separate path can
   diverge and reintroduce the confound above.
3. **Total sample size `n = G · m` is held fixed** within a comparison slice. A
   fixed-`G` slice also exists, but the two must never be mixed in one figure.
4. **Analytic confidence intervals attach only to the estimator they were derived
   for.** See §4. This has bitten us once already.
5. **Seeds are never traded away for grid size.** If compute is short, cut the grid.
   Error bars are the deliverable.

---

## 3. The generators (exact statements)

Dependence is imposed in **latent space**, then mapped to ambient space by an
embedding. This is what makes exact marginal preservation possible.

### Grouped / equicorrelated (primary condition)

For group `g` and member `j`, with `u_g, e_{g,j} ~ iid N(0, I_d)`:

```
z_{g,j} = sqrt(rho) * u_g + sqrt(1 - rho) * e_{g,j}
```

Properties, all exact for every `rho` in [0, 1):

- `z_{g,j} ~ N(0, I_d)` — a sum of independent Gaussians is Gaussian, and
  `rho * I + (1 - rho) * I = I`. **The marginal does not depend on rho.**
- `Cov(z_{g,j}, z_{g,j'}) = rho * I_d` for `j != j'` in the same group.
- `Cov = 0` across groups.
- Intraclass correlation is exactly `rho`.
- `rho = 0` reduces to i.i.d. sampling.

### Sequential / AR(1) (secondary condition)

Structured as **`G` independent stationary chains of length `m`** — not one chain of
length `n`. This keeps the block structure identical to the grouped arm, so the two
arms differ in exactly one thing: the within-block correlation *pattern*. It also
keeps leave-one-group-out jackknife well defined.

Within each chain:

```
z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t,   eps_t ~ iid N(0, I_d)
```

With `z_0 ~ N(0, I_d)` the chain is stationary with marginal **exactly** `N(0, I_d)`.
Initializing `z_0` any other way introduces a burn-in transient — don't.

**`rho` does not mean the same thing across the two arms.** Equicorrelated: every
within-block pair correlates at `rho`. AR(1): a pair at lag `k` correlates at
`rho^k`, so average within-block dependence is strictly lower at the same nominal
`rho`. Never plot the two arms against a shared `rho` axis as though it were a matched
comparison — part of any apparent difference is simply less dependence. Compare each
arm against **its own `rho = 0`**, and record average pairwise within-block
correlation as a derived column so a matched-dependence comparison stays possible
later.

### Non-Gaussian marginals

Apply a Gaussian copula: transform componentwise by `F^{-1}(Phi(z))` for the target
marginal `F`. This preserves the rank dependence structure and gives the exact target
marginal.

### Embeddings

Latent `z` in `R^d` is mapped to ambient `R^D`:

- **Linear** — random orthonormal `D x d` map. Here the true ID is exactly `d`.
- **Smooth nonlinear** — a **graph embedding**, specified so that the true intrinsic
  dimension is guaranteed to remain `d`:

  ```
  phi(z) = R @ concat[ z , g(z) ]
  ```

  where `g: R^d -> R^(D-d)` is smooth and `R` is a random `D x D` orthogonal matrix.
  Default `g_j(z) = a_j * sin(omega * dot(b_j, z) + c_j)` with random unit `b_j` and
  phases `c_j`. `omega` is the **curvature knob**; `omega -> 0` recovers the linear
  case.

  Why this construction and not a hand-picked curved map:

  - **Full-rank Jacobian, always.** `D phi = R @ [ I_d ; dg/dz ]`. The identity block
    forces rank `d` for *every* choice of `g` and *every* `z`. No dimensional collapse
    is possible.
  - **Injective, always.** The first `d` coordinates of `R^T phi(z)` recover `z`
    exactly, so `phi` cannot self-intersect.
  - `R` is a diffeomorphism, so it preserves intrinsic dimension while removing
    axis-alignment artifacts.

  Both properties hold **by construction**, not by inspection. The numerical checks
  below are therefore regression guards against implementation bugs, not the source of
  the guarantee.

  Required checks (`tests/test_embeddings.py`):
  - Numerical Jacobian at many sampled `z`: singular values via SVD, assert exactly
    `d` above tolerance and a clear spectral gap.
  - No dimensional collapse: local PCA on small neighbourhoods shows `d` dominant
    components.
  - Bi-Lipschitz sanity on the sampled region: the ratio
    `||phi(z1) - phi(z2)|| / ||z1 - z2||` stays bounded away from `0` and infinity
    across many random pairs. Guards against a curvature setting so aggressive that
    the manifold is numerically indistinguishable from a lower-dimensional set at the
    sampled density.
  - Sweeping `omega -> 0` reproduces the linear embedding's estimates.

  **Embedding parameters are frozen within every comparison slice.** The orthogonal
  matrix `R`, the `b_j`, amplitudes `a_j`, phases `c_j`, and curvature `omega` must be
  **identical across all `rho` and `m` conditions being compared**. If the embedding
  were redrawn per condition, the manifold itself would vary with the treatment and
  every comparison would be confounded.

  Implement by drawing all embedding parameters from a dedicated **`embedding_seed`**,
  separate from the data seed. Record `embedding_seed` and the full embedding config
  (`d`, `D`, `omega`, kind) on **every** run row. The current plan uses a single
  `embedding_seed` per slice; robustness across multiple embedding realizations is
  Future work.

  See also the participation-ratio caveat in §6.

---

## 4. The estimators, and which interval belongs to which

There are **two distinct estimators both called "TwoNN"** in the literature. They are
not interchangeable. The plan includes and labels both.

Both start from the same model. For each point, let `r_1` and `r_2` be the distances
to its first and second nearest neighbours, and `mu = r_2 / r_1`. Under the model,
`mu ~ Pareto(1, d)` with density `d * mu^{-(d+1)}` for `mu >= 1`, so
`log(mu) ~ Exponential(rate = d)`.

### Which outcomes apply to which estimator

| Outcome | Applies to |
|---|---|
| Bias, variance, RMSE, failure rate | The **three nearest-neighbour estimators**: `TwoNN_Regression`, `TwoNN_MLE`, and `LevinaBickel` |
| Participation-ratio bias and RMSE against true `d` | **Isotropic linear Gaussian embedding only** |
| Participation ratio on nonlinear embeddings | Change relative to its matched **`rho = 0, m = 1` baseline**; never bias or RMSE against true `d` |
| Analytic interval coverage | **`TwoNN_MLE` only** — the interval is derived from its likelihood and does not exist for the others |
| Jackknife interval coverage (Day 4) | **`TwoNN_MLE` only** in the current project scope |

The jackknife is generic and *could* wrap any estimator. Restricting it to
`TwoNN_MLE` is a deliberate scope decision, not a mathematical necessity: it keeps
Day 4 cost to a quarter and keeps analytic-vs-jackknife a like-for-like comparison on
one estimator. Extending jackknife inference to the others is **Future work** — do not
expand the current project scope.

Consequence for the cache: the Day 4 jackknife run computes only `TwoNN_MLE`, which
needs 2 neighbours, so **`k_max = 2` and `K = m + 2`** there. Do not carry over
Levina–Bickel's `k` from the Stage A runs.

### (a) `TwoNN_Regression` — Facco et al. 2017

Fits a line through the origin of `-log(1 - F_emp(mu))` against `log(mu)` and takes the
slope, after **discarding the largest 10% of `mu` values**. This is what
`skdim.id.TwoNN` implements (verified against skdim 0.3.4 source).

**To be implemented in-repo from the shared `r_1, r_2` distances — do not call
`skdim.id.TwoNN` inside the experiment loop.** skdim computes its own distances
internally, so calling it would contradict the shared-kNN design and silently double
the neighbour work. Replicate its convention exactly:

```
mu    = r_2 / r_1                       for all N points
keep  = int(N * (1 - discard_fraction)) smallest mu, ascending
Femp  = arange(keep) / N                # denominator N, NOT keep
slope = lstsq( log(mu_kept) , -log(1 - Femp) )   # through the origin
```

Note `Femp` divides by `N`, not by the truncated count, and starts at `0`. Getting
that wrong shifts the estimate slightly and would look like a real effect.
`tests/test_estimators.py` must assert agreement with `skdim.id.TwoNN` to tight
tolerance under identical settings and truncation.

**No analytic interval exists for this estimator.** Two independent reasons: its
sampling distribution is not the MLE's, and the 10% truncation means it isn't even
fitting the untruncated model. Do not attach the interval from (b) to it.

### (b) `TwoNN_MLE` — Denti et al. 2022

The maximum-likelihood estimator for the same model:

```
d_hat = n / sum(log(mu_i))
```

Let `S = sum(log(mu_i))`. Treating `log(mu_i) ~ iid Exp(d)` gives `S ~ Gamma(n, rate=d)`
and therefore `2 * d * S ~ ChiSquare(2n)`, yielding the `(1 - alpha)` interval:

```
[ chi2.ppf(alpha/2, 2n) / (2S),  chi2.ppf(1 - alpha/2, 2n) / (2S) ]
```

**Read the exactness claim carefully.** This interval is exact *only under the
idealized model in which the ratios `mu_i` are independent Pareto variables*. It is
**not** exact merely because the underlying points were sampled independently. The
`mu_i` are dependent even for i.i.d. points, because neighbouring points share
neighbours — the internal dependence Denti et al. explicitly note. So the product
`prod_i d * mu_i^{-(d+1)}` is a **pseudo-likelihood**, `d_hat` is a pseudo-MLE, and
the interval carries whatever miscoverage that idealization costs, before clustering
enters at all.

Three consequences, all load-bearing:

1. **The analytic interval may already under-cover at `rho = 0`.** That is expected
   and is not our finding.
2. Therefore in every point-cloud experiment, **report both numbers**:
   - **Absolute coverage** — the raw fraction of intervals containing the truth.
     Answers "is this interval usable at all?" If absolute coverage is 40%, that
     matters regardless of what the control did.
   - **Coverage change relative to the `rho = 0` control** — this is the number that
     carries the **causal attribution to clustering**. Reporting only the absolute
     shortfall would let a reviewer attribute the whole effect to the shared-neighbour
     issue Denti already described.

   Neither alone is sufficient. Relative change isolates our effect; absolute coverage
   establishes practical usability.
3. A separate **i.i.d.-Pareto sanity check** is required: simulate `mu_i` directly as
   i.i.d. Pareto(1, d), build the interval, confirm coverage hits nominal. This
   validates the interval *machinery* independently of any geometry, isolating a third
   possible failure source (bugs in the chi-square algebra) from the other two.

Must be validated numerically against the R package `intRinsic` before use, with
**trimming and bias-correction settings explicitly matched on both sides**. `intRinsic`
trims by default, and there is a convention question on bias correction: since
`S ~ Gamma(n, d)`, `E[n/S] = n*d/(n-1)`, so the plain MLE is biased upward and the
unbiased form is `(n-1)/S`. Validating an untrimmed, uncorrected implementation
against a trimmed, corrected reference produces a mismatch that invites "fixing" the
wrong side.

### (c) `LevinaBickel` — MLE over `k` nearest neighbours

Uses the full local neighbourhood rather than just the first two. Record which
averaging convention is used (average of estimates vs. average of inverses — the
MacKay–Ghahramani correction), because published numbers differ on this.

### (d) `ParticipationRatio` — non-nearest-neighbour comparator

`PR = (sum of covariance eigenvalues)^2 / (sum of squared covariance eigenvalues)`.

Included deliberately as a **mechanistic contrast**: it is affected by grouping through
the covariance matrix rather than through neighbour distances. See §6 for how to score
it. Bias and RMSE against true `d` are meaningful only for the isotropic linear
Gaussian embedding. On nonlinear embeddings, compare participation ratio with its own
matched `rho = 0, m = 1` baseline.

---

## 5. Uncertainty methods under test

| Method | Notes |
|---|---|
| Analytic (Denti) | Exact **only under the idealized i.i.d.-Pareto ratio model** (§4b) — *not* merely when the points were sampled independently. `TwoNN_MLE` only. Free to compute. |
| Leave-one-group-out jackknife | Primary cluster-aware method. Full construction below. |

### Jackknife interval — complete specification

A variance estimate is not an interval. Coverage is undefined until centre, standard
error, and critical value are all pinned. For `G` groups, let `theta_hat` be the
full-sample estimate and `theta_hat_(g)` the estimate with group `g` removed:

```
theta_bar = (1/G) * sum_g theta_hat_(g)
Var_jack  = ((G - 1) / G) * sum_g (theta_hat_(g) - theta_bar)^2
SE        = sqrt(Var_jack)

CI_95 = theta_hat  +/-  t_{G-1, 0.975} * SE
```

- **Centre: `theta_hat`**, the full-sample estimate — *not* `theta_bar`, and no
  bias-corrected pseudo-value form. This is the standard survey-statistics convention.
- **Critical value: `t_{G-1}` is the pre-registered candidate, and is itself under
  evaluation.** The `t_{G-1}` convention comes from normal-theory linear models, where
  a variance estimated from `G` replicates carries roughly `G - 1` degrees of freedom.
  **That justification is not guaranteed to transfer** to a nonlinear, non-smooth
  estimator under clustered sampling — which is the whole reason Day 4 measures
  coverage instead of asserting it. We choose `t_{G-1}` because it is the conventional
  survey-statistics choice and is the wider, more conservative of the two.

  **Report both `t_{G-1}` and `z` coverage.** They are computed from identical
  replicates at zero extra cost, so the choice can be settled empirically rather than
  argued. The two differ materially only at small `G`: at `m = 50, n = 1500` we have
  `G = 30`, where `t_29 = 2.0452` against `z = 1.9600`, a 4.35% wider interval; at
  `m = 10` (`G = 150`) the gap is 0.8%. Do not claim in advance that either choice is
  correct — that is a finding, not an assumption.

**Secondary variant, log scale (free).** `d_hat` is positive and right-skewed, so a
symmetric interval can emit a negative lower bound at small `G`. Build the same
interval on `log(theta_hat)` and exponentiate. Costs **zero additional estimator
recomputations** — identical replicates, different transform. If better calibrated,
"use the log scale" is an actionable recommendation. Drop this first if Day 4 overruns.

**Failure handling.** If any leave-one-group-out replicate fails (NaN, non-positive,
exception), mark the **whole trial** as failed. Never compute the variance over the
surviving replicates — silently dropping failures biases the variance estimate
downward, which is indistinguishable from the overconfidence this project is trying
to detect.

**Implementation note — do not recompute kNN per replicate.** The naive reading of "G
recomputations" is `O(n^2 * G)` per trial and will blow the Day 4 timebox. Instead
compute each point's sorted neighbour list **once per trial**, retain the top

```
K = m + k_max
```

where **`k_max` is the largest neighbourhood size any nearest-neighbour estimator in
the run requires** (TwoNN needs 2; Levina–Bickel needs its `k`). For each removed
group, walk down each retained point's cached list skipping deleted points. That is
`O(n * K)` per replicate.

Why `m + k_max` exactly: a single removed group deletes at most `m` points, so a
retained point loses at most `m` cached neighbours and **is guaranteed to retain at
least `k_max`**. The buffer is tight — no arbitrary constant.

**Fallback semantics.** Fall back to full recomputation whenever any retained point
has fewer than `k_max` surviving neighbours — not merely when its whole cached list is
deleted. Under equal group sizes this condition is **provably unreachable**, so the
fallback is an **assertion that signals a bug** (or unequal group sizes) rather than
an expected code path. If it fires during a run, stop and investigate; do not treat it
as normal.

**`k_max` must be recorded on every run.** Changing Levina–Bickel's `k` silently
changes the required cache depth, and a stale `K` would corrupt results without
raising an error.

The cached path must be verified **numerically identical** to naive full recomputation
on a small case before it is trusted (Day 1).

**Jackknife is under test, not assumed correct.** The delete-one-cluster jackknife is
standard in survey statistics, but its validity is not guaranteed here: the jackknife
is known to fail for **non-smooth** statistics, and nearest-neighbour distances are
non-smooth — the identity of a point's nearest neighbour changes discontinuously as
the sample changes. So Day 4 reports *how these interval methods actually behave under
clustering*. It does not present jackknife as a proven fix. If jackknife coverage is
also poor, that is a legitimate and useful result.

**Do not use bootstrap with replacement.** Resampling points or groups with
replacement duplicates vectors, so some point acquires a duplicate at distance zero,
giving `r_1 = 0`. Then `mu = r_2 / r_1` diverges and `log(r_k / r_j)` hits `log(0)`.
This is a hard failure specific to nearest-neighbour estimators, not a mild bias.

**Group subsampling is deliberately excluded — see ROADMAP Future work.** Subsampling
inference (Politis–Romano) requires knowing the estimator's convergence rate in order
to rescale subsample estimates. Under grouped sampling that rate is governed by how
fast information accumulates — which is exactly the effective sample size this project
sets out to measure. Choosing a scaling rule would mean assuming the answer to the
research question. Estimating the rate empirically across subsample sizes is a possible
escape, but it is a separate project, not a one-week task.

---

## 6. Known traps

Every one of these was found the hard way or caught in review. Do not undo them.

- **`skdim.id.TwoNN` is not the MLE.** §4. Never attach the analytic interval to it.
- **The analytic interval is not exact just because points were sampled
  independently.** §4b. It is exact only under the idealized i.i.d.-Pareto model for
  the ratios. Coverage is therefore always reported **both ways** — absolute, and as
  change relative to the `rho = 0` control — with the relative figure carrying the
  causal claim. Reporting only one is a correctness bug: relative alone hides whether
  the interval is usable, absolute alone misattributes Denti's known effect to us.
- **Never write "same data, different grouping."** §2, invariant 1. The one-point
  marginal is preserved; the joint distribution and the pairwise geometry are not, by
  design.
- **`rho` is not comparable across the grouped and AR(1) arms.** §3.
- **Bootstrap with replacement breaks NN estimators.** §5.
- **KS tests cannot validate the generators.** The KS test assumes i.i.d. samples, so
  its null distribution is invalid on exactly the dependent data we need to check.
  Marginal preservation is guaranteed by the algebra in §3; tests should check
  **moments, quantiles, variance, and empirical ICC against tolerances**, and treat
  any distributional test as a smoke check only.
- **Participation ratio has no guaranteed relationship to `d` on curved manifolds —
  in either direction.** Do not claim `PR > d`. `PR = (sum lambda)^2 / sum lambda^2`
  is maximised at `d` when the `d` eigenvalues are equal; unequal eigenvalues push it
  **below** `d`, while curvature adds extra small eigenvalues pushing it up. The net
  direction is not determined. PR is a global spectral quantity, not a manifold
  dimension estimator. Bias and RMSE against true `d` are valid only for the isotropic
  linear Gaussian embedding. On nonlinear embeddings, score it **relative to its own
  matched `rho = 0, m = 1` baseline**, never as bias or RMSE against true `d`. Mixing
  conventions across estimators in one table is a correctness bug, not a presentation
  choice.
- **Kish's design effect applies to the linear Gaussian grouped case only.**
  `n_eff = n / (1 + (m - 1) * rho)` is derived for the variance of a *sample mean*
  under equal-size equicorrelated clusters. `rho` is defined in latent space; a
  nonlinear embedding does not preserve equicorrelation-with-parameter-`rho` in
  ambient space, and transformer activations have no `rho` at all. Plot it as a
  reference line on the one panel where it is legitimate. Never as ground truth.
- **Grouped sampling is not exotic.** Independent groups of fixed size are ordinary
  finite-range dependence and are formally covered by existing mixing theory. Do not
  claim otherwise in any writeup. The contribution is finite-sample estimator and
  interval behaviour, which that theory does not address.
- **Variance ratios need seeds.** At 40 seeds a variance estimate carries ~23%
  relative standard error and a *ratio* of two carries ~32% — too noisy to resolve the
  effects the study targets. Stage A will run at 150 seeds.
- **Stage A thresholds cannot be tuned after seeing Stage A.** They are preregistered
  in the final section of this file. Revising practical-effect cutoffs, confidence
  criteria, the primary panel, or the coherence rule after observing results would
  invalidate the go/no-go decision.

---

## 7. Conventions

- Follow the repository-wide safety rules in [AGENTS.md](AGENTS.md) and the reporting,
  secret-handling, dependency, and untrusted-input guidance in
  [SECURITY.md](SECURITY.md). The checked-in files are authoritative; local shared
  context may supplement them but is not required.
- Python 3.12, `uv` for everything. `uv sync`, `uv run <cmd>`. Never `pip install`
  into the environment directly.
- `numba>=0.60` is pinned explicitly. `scikit-dimension`'s sdist metadata otherwise
  resolves to numba 0.53.1, which caps at Python 3.9 and fails to build. Do not
  remove that pin.
- Planned library code belongs in `src/csid/`; planned runnable scripts belong in
  `experiments/`.
- Every future experiment script must write a **parquet of raw per-run rows** to
  `results/raw/`, and figures must be built from that file in a separate step. Never
  plot straight from an in-memory run — results must be reproducible without
  recomputing.
- Every run must record its seed. Every figure must record the git SHA that produced
  it.

### Result storage policy (one rule, no exceptions)

| Location | Contents | Committed? |
|---|---|---|
| `results/raw/` | Full per-run rows, one row per (condition, seed). Large, regenerable from the script plus its seed. | **No** — gitignored |
| `results/summaries/` | Compact per-condition aggregates (`.csv` or `.parquet`) that the report and figures actually consume. Small. | **Yes** — committed |
| `docs/figures/` | Figures cited by the report. | **Yes** — committed |

Every figure must be reproducible from a **committed summary file** alone, with no
access to `results/raw/`. That is what makes the repository self-contained for a
reader. Raw files stay local because they are large and fully regenerable.
- Planned tests must be property tests on the generators, not snapshot tests on
  numbers.

## 8. Scope discipline

Out of scope, permanently, unless the roadmap is formally revised: unions of
manifolds, additional estimators beyond the four in §4, a second transformer,
GPU-scale experiments, and any claim about the true ID of a real neural
representation.

If a result looks exciting and suggests a new experiment, write it in the "Future
work" section of ROADMAP.md rather than starting it.

## Stage A decision thresholds

These thresholds are preregistered before Stage A results are observed. They must not
be revised after seeing those results.

### Decision panel

The primary go/no-go panel is the **grouped-dependence, linear Gaussian embedding**.
Each dependent condition is compared with its matched `rho = 0` control. AR(1),
nonlinear embeddings, and transformer results are secondary analyses for this
decision. Under the outcome scope in §4, coverage here means `TwoNN_MLE` coverage,
while bias and variance are evaluated separately for each of the three
nearest-neighbour estimators. Participation ratio is reported with its
embedding-specific scoring but is outside the three-signal null gate.

For a dependent condition minus its matched control:

- **Coverage:** a practically meaningful degradation requires an absolute decrease of
  at least **5 percentage points** and a 95% Monte Carlo confidence interval for the
  coverage difference that excludes zero.
- **Bias:** a practically meaningful change requires
  `abs(delta_bias) / d >= 0.05` and a 95% Monte Carlo confidence interval for the bias
  change that excludes zero.
- **Variance:** a practically meaningful change requires a dependent/control variance
  ratio of at least **1.25** or at most **0.80**, and a 95% Monte Carlo confidence
  interval for the ratio that excludes 1.

### Coherence and control checks

A single isolated grid cell cannot determine the decision. At least one meaningful
effect must form a coherent pattern across two or more neighbouring dependence
conditions, such as increasing `rho` or increasing `m`.

- If matched `rho = 0` coverage is at or below **20%**, label the contrast
  **floor-limited**.
- If matched `rho = 0` coverage is at or above **99%**, label the contrast
  **ceiling-limited**.
- Report floor- and ceiling-limited contrasts, but do not use either as the sole
  evidence for the Stage A go/no-go decision.
- If the matched control has `abs(bias) / d >= 0.20`, label it **substantially
  control-biased** and interpret its coverage result as an estimator-baseline
  limitation.

### Null-result stopping rule

Stage A supports a null-result stop only when **none** of coverage, bias, or variance
meets its threshold as a coherent pattern in the primary decision panel. Failure rates
must always be reported. Failure rate is not part of the three-signal null gate unless
failures prevent the estimator from being meaningfully evaluated.
