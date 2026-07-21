# Clustered Sampling and Intrinsic-Dimension Estimation

**Status:** Planning only. No implementation or experiments have been completed.

> **New here?** Read **[ELI5.md](ELI5.md)** — the full plain-language explanation, no
> maths background assumed. It also has a suggested reading path into the literature.

A small, self-contained empirical study asking whether a widely used measurement
tool still works when its main assumption is quietly violated — and whether the
error bars people report alongside it mean what they claim.

---

## TL;DR

Intrinsic-dimension (ID) estimators are used across machine learning to ask "how
many numbers do you *really* need to describe this data?" Nearly all of them assume
data points are **independent draws**. In practice they get applied to neural network
activations sampled as **many tokens from the same few sentences** — which are not
independent.

This is textbook *clustered sampling*. Its classical consequence is that your
**effective** sample size is smaller than your actual sample size, so your error
bars should be wider than you think.

The study will measure what clustering does to four ID estimators — three families,
with two distinct TwoNN variants that are routinely conflated. The three
nearest-neighbour estimators will be evaluated for bias, variance, RMSE, and failure
rate; confidence-interval coverage is the primary outcome and applies only to the
likelihood-based `TwoNN_MLE`.

---

## Explain it like I'm 5

### What is "intrinsic dimension"?

Imagine a crumpled piece of paper sitting in a room.

To say where a point on that paper is, you *could* use 3 numbers — left/right,
forward/back, up/down. But the paper is really just a flat sheet. If you uncrumpled
it, 2 numbers would be enough.

So: the paper lives in **3** dimensions, but its **intrinsic dimension** is **2**.

Real datasets are like this. A dataset might have 1,000 columns, but really only
vary in ~10 meaningful ways. That "10" is the intrinsic dimension, and knowing it
tells you something real about the data — or about a neural network's internals.

### How do you measure it?

The common tools work by looking at **how far each point is from its nearest
neighbors**.

The intuition: in a low-dimensional space, points have close neighbors. In a
high-dimensional space, everything is far from everything else — there's too much
room. So the pattern of neighbor distances leaks information about the true
dimension, and you can work backwards from it.

### What's the problem?

These tools assume every data point is an **independent** draw. Like pulling marbles
out of a bag, shaking the bag between each pull.

But here's how people actually use them on language models: they run some sentences
through the model and grab the internal activations for **many tokens from each
sentence**. Those points are *not* independent. Tokens from the same sentence are
about the same topic, in the same style, in the same context. They resemble each
other.

**The analogy:** you want to know the average height in a city. You pick 20
households and measure everyone in each. You now have 100 measurements — but you do
*not* have 100 independent pieces of information, because family members resemble
each other. Your real information content is somewhere closer to 20.

Statisticians call this **clustered sampling**, and the well-known consequence is
that your **effective sample size** is smaller than your raw count. If you ignore
this, your error bars come out too narrow and you get more confident than the data
justifies.

### So what's the question?

> When you feed clustered data to intrinsic-dimension estimators, what happens?

Three sub-questions:

1. **Bias** — does the estimate drift away from the true answer?
2. **Variance** — does the estimate get noisier?
3. **Coverage** — when a method reports a "95% confidence interval," does that
   interval actually contain the true answer 95% of the time?

Number 3 is the one we care about most, and here's why: clustering reduces how much
*information* you have without reducing how many *points* you have. A method that
counts points to decide how confident to be will therefore be **overconfident** — its
interval can be far too narrow *even when the point estimate itself looks fine*.

### Why does this matter?

Papers report things like "layer 12 of this model has intrinsic dimension 15" and
build arguments on top of that number. If the number moves depending on how you
sampled your tokens, or if the error bars around it are too narrow to be honest,
those arguments are shakier than they look.

This is a *check the ruler is straight before you measure with it* project.

---

## The hypothesis

For the nearest-neighbour estimators and `TwoNN_MLE` coverage, we predict in rough
order of confidence:

1. **Coverage degrades as clustering increases.** *(Most confident.)* Reported **both
   in absolute terms and as change against the `ρ = 0` control** — see the note below
   on why both numbers are needed.
2. **Variance increases.** More clustering → noisier estimates at the same raw
   sample size.
3. **Bias may shift**, but possibly not much. It is entirely possible the point
   estimate is nearly fine while the interval around it is badly wrong. That
   combination would itself be the interesting result.

We deliberately **do not** predict a specific effective sample size in advance. The
study will estimate it empirically (see below).

---

## The experiment

### Step 1 — Build fake data where we know the true answer

You cannot check a ruler without something of known length. The study will generate
synthetic data on shapes whose dimension is chosen in advance.

It will sample points in two ways:

- **Independent** (the control) — every point drawn on its own.
- **Clustered** — points come in groups, and points within a group resemble each
  other. A dial `ρ` (rho) controls *how much* they resemble each other: `ρ = 0` means
  not at all, `ρ = 0.9` means a lot.

**The critical control:** the construction is arranged so that **any single point,
looked at on its own, comes from exactly the same distribution regardless of `ρ`**.
Same shape, same spread, same everything — for one point in isolation.

What *does* change is how points relate to **each other**. Points in the same group
sit closer together, and the more you turn up `ρ`, the closer they sit.

That is not a flaw in the experiment. It is the entire mechanism the study will test:
because groupmates are closer, they're more likely to end up as each other's nearest
neighbours, and the estimator reads those shortened distances as evidence about
dimension. So the honest description is **"same one-point distribution, different
relationships between points"** — not "same data, different labels." Getting this
distinction right matters, because otherwise you'd reasonably wonder how the estimate
could move at all.

The control still does its job: since one-point behaviour is pinned, any observed
difference must come from the relationships between points, which is exactly the
quantity changed by the dependence dial.

### Step 2 — Use estimator-specific outcomes

The study will run four ID estimators — three families, with two distinct TwoNN
variants:

| Estimator | Family |
|---|---|
| TwoNN (regression fit, Facco et al.) | nearest-neighbour |
| TwoNN (likelihood MLE, Denti et al.) | nearest-neighbour |
| Levina–Bickel | nearest-neighbour |
| Participation ratio | covariance spectrum (non-neighbour contrast) |

The two TwoNN variants are different estimators that the literature often treats as
one. The plan carries both separately so any different response to clustering remains
visible.

Outcome scope is estimator-specific:

| Outcome | Applies to |
|---|---|
| Bias, variance, RMSE, and failure rate | The three nearest-neighbour estimators |
| Participation-ratio bias and RMSE against true `d` | Isotropic linear Gaussian embedding only |
| Participation ratio on nonlinear embeddings | Change relative to its matched `rho = 0, m = 1` baseline; never bias against true `d` |
| **Analytic interval coverage** | `TwoNN_MLE` only |
| **Jackknife interval coverage** | `TwoNN_MLE` only in the current project scope |

#### Why coverage is reported two ways

The interval the study will test is exact under an *idealized* model — one where the
quantities it's built from behave like clean independent draws. They don't, quite,
even when the data points themselves are independent, because neighbouring points
share neighbours. That's a known wrinkle, already documented by the authors of the
interval.

So the interval may come in slightly under 95% **even in our control condition**, with
no clustering at all. If we reported that as our finding, we'd be taking credit for
somebody else's known result.

The study will therefore report **both** numbers, because each answers a different
question:

- **Change against our own control** — if coverage is 91% with no clustering and 74%
  with heavy clustering, the clustering cost is the drop from 91 to 74. That part is
  ours, and it's the number that supports a causal claim.
- **Absolute coverage** — 74% is 74% regardless of what the control did. If an
  interval advertising 95% delivers 74%, it is not usable, and a reader needs to know
  that directly rather than reconstruct it from a delta.

Reporting only the relative change would hide whether the tool is usable; reporting
only the absolute number would let us take credit for someone else's known result. The
plan also includes a separate check on idealized data where the interval *should* hit
95% exactly, to confirm the interval calculation is correct. Three possible sources of
error, each isolated.

### Step 3 — Ask how much information was really there

The payoff question:

> Clustered data with 2,500 points behaves like independent data with **how many**
> points?

The study will answer it by measurement, not assumption: it will first chart how the
estimators behave on *independent* data as the sample size grows, then look up where
each clustered result lands on that chart. That lookup value will be the **implied
effective sample size**.

There's a classic formula from survey statistics (Kish's design effect) that predicts
this for simple averages. The study will plot it as a **reference line** on the one
experiment where it legitimately applies — to see whether these more complicated
estimators follow it or depart from it. Either answer is informative.

### Step 4 — Check it against a real model

Finally, the plan includes a small transformer on CPU. The study will hold the **total
number of activation vectors fixed** and vary how they are collected — many sentences
with one token each, versus few sentences with many tokens each, and several protocols
in between.

It will then report how much the answer moves.

**The study will make no claim about the true intrinsic dimension of a real neural
network.** This step is planned to measure *sensitivity to an arbitrary choice that
papers don't usually report* — nothing more.

---

## What this project does NOT claim

Stated up front, because scope discipline is most of what makes a small result
trustworthy:

- We do **not** claim ID estimators are broken in general. They are well-founded
  under their stated assumptions. The planned study will test what happens outside
  them.
- We do **not** claim to know the true intrinsic dimension of any real neural
  representation.
- We do **not** claim clustered sampling is mathematically exotic. Independent
  groups of fixed size are ordinary finite-range dependence, and existing theory
  formally covers them. The planned contribution is *finite-sample behavior of the
  estimators and their intervals*, which that theory does not address.
- Kish's design effect will be used as a **reference curve for one specific linear
  Gaussian experiment only** — not as ground truth for curved shapes or for real
  transformer activations.
- A null result is a real outcome here. If clustering turns out not to matter at
  reachable scale, it will be reported plainly.

---

## Related work

### Methods we build on

The estimators to be tested are not ours. In order of how central they are here:

- **TwoNN** — [Facco, d'Errico, Rodriguez & Laio (2017)](https://www.nature.com/articles/s41598-017-11873-y),
  *Estimating the intrinsic dimension of datasets by a minimal neighborhood
  information*. The estimator this project is mostly about. Note that "TwoNN" is
  ambiguous in practice: this paper's empirical-CDF regression fit and the
  likelihood-based MLE below are **different estimators**, and the plan carries both,
  separately labelled.
- **Levina–Bickel MLE** — Levina & Bickel (2004), *Maximum Likelihood Estimation of
  Intrinsic Dimension*, NeurIPS. Same family, uses more than two neighbours.
- **`scikit-dimension`** — [Bac, Mirkes, Gorban, Tyukin & Zinovyev (2021)](https://www.mdpi.com/1099-4300/23/10/1368),
  *Entropy* 23(10):1368. The library we plan to use for the standard estimators, so that
  a botched re-implementation can't be blamed for any result.

### Where this sits in the literature

Three papers bound the planned contribution. Each is close, and each leaves the
specific gap the study targets. A fourth recent estimator is adjacent to the methods
under study. **These are chosen for defending novelty, not for learning the material**
— if you want to understand the field, see the reading path in
[ELI5.md](ELI5.md#what-to-read-in-what-order) instead.

**[Nearest-Neighbor Radii under Dependent Sampling](https://arxiv.org/abs/2605.14343)**
— Gao, Hou & Lin (2026). The closest work on dependence. They prove that the
*geometric primitive* underneath these estimators — the distance to your k-th nearest
neighbor — stays well-behaved under dependence, holding the marginal distribution
fixed exactly as the study plans to do. Their finding is that dependence affects
constants rather than asymptotic scaling.
**Gap:** they analyze the radius itself. The planned study will analyze the nonlinear
estimators built on top of it and the uncertainty intervals — neither of which they
touch. Constants are exactly what matters at finite sample sizes.

**[The generalized ratios intrinsic dimension estimator](https://www.nature.com/articles/s41598-022-20991-1)**
— Denti, Doimo, Laio & Mira (2022). The inference foundation. They derive
likelihood-based confidence intervals for the TwoNN estimator, and they explicitly note
that nearest-neighbor ratio observations are not always independent, because nearby
points can share neighbors.
**Gap:** that is *internal* dependence arising from the geometry. The planned study
will examine *externally imposed* dependence from how the data was collected — a
separate source, and the one practitioners actually control.

**[Rethinking Intrinsic Dimension Estimation in Neural Representations](https://arxiv.org/abs/2604.20276)**
— Schulte & Rügamer (AISTATS 2026). Broader context for neural ID. Their critique is
topological: token representations form a finite, separated set rather than a smooth
manifold, so the manifold assumption fails at the root.
**Gap:** complementary to ours, and a reason the planned transformer step will report
*sensitivity only* and make no claim about a true dimension.

**[A Universal Nearest-Neighbor Estimator for Intrinsic Dimensionality](https://arxiv.org/abs/2603.10493)**
— Ong, Bobrowski, Reinert & Skraba (2026). This recent work proposes L2N2, an estimator
based on nearest-neighbour distance ratios, and proves a universality result with
respect to the distribution generating the data. It is adjacent because it belongs to
the same broad nearest-neighbour estimator family. This repository does not rely on it
for any claim about clustered dependence or confidence-interval coverage, and L2N2 is
not part of the current experimental grid.

### Motivating and background work

- [Ansuini, Laio, Macke & Zoccolan (NeurIPS 2019)](https://arxiv.org/abs/1905.12784),
  *Intrinsic dimension of data representations in deep neural networks* — the paper
  that popularised measuring ID layer-by-layer inside networks. The methodology the
  study plans to stress-test is largely the one this work established.
- [Binnie, Dłotko, Harvey, Malinowski & Yim (2025)](https://arxiv.org/abs/2507.13887),
  *A Survey of Dimension Estimation Methods* — broad survey of the estimator zoo, with
  an evaluation of how methods respond to curvature and noise. Its stated motivation
  is close to ours: many estimators exist, little guidance on using them reliably.
- Kish, *Survey Sampling* (1965) — origin of the design effect and effective sample
  size, the classical treatment of the clustered-sampling problem used by the plan.

---

## Planned development setup

The planned implementation requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd clustered-sampling-id
uv sync
uv run pytest  # available after Day 1 adds the test suite
```

## Planned repository layout

At present, the repository contains planning documents, dependency metadata, an empty
`src/csid/__init__.py` package marker, and `docs/repo-context.md`. The `tests/`,
`experiments/`, and `results/` directories do not exist yet. The report and figure
subpaths under `docs/` are also planned rather than implemented.

```
src/csid/           empty package scaffold; planned library code will live here
tests/              planned property and validation tests (does not exist yet)
experiments/        planned stage-specific scripts (does not exist yet)
results/raw/        planned local per-run output (gitignored; does not exist yet)
results/summaries/  planned committed aggregates (does not exist yet)
docs/repo-context.md current concise repository context
docs/REPORT.md      planned final writeup
docs/figures/       planned committed report figures
ELI5.md             plain-language explanation and reading path
ROADMAP.md          plan, progress, and stopping criteria
CLAUDE.md           durable technical design context
```

## License

MIT. See [LICENSE](LICENSE).
