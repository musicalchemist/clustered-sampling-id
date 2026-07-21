# Explain it like I'm 5

The long, plain-language version. No maths background assumed. If you read only one
document in this repo, read this one.

The [README](README.md) has the condensed version. [CLAUDE.md](CLAUDE.md) has the
precise technical statements. This file is the bridge between them.

---

## The one-paragraph version

There's a measuring tool that machine-learning researchers use a lot. It has a rule
for when it works: your data points have to be collected independently of each other.
In practice, people routinely break that rule. We went looking for someone who had
measured what that costs, and couldn't find it. The planned study will measure that
cost, focusing on whether the **error bars** that come with the tool still mean
anything once the rule is broken.

---

## Part 1 — What is "intrinsic dimension"?

### The crumpled paper

Take a sheet of paper and crumple it into a ball. Now it's sitting on your desk.

If I ask "where is this particular ink dot?", you need **3** numbers: how far left,
how far forward, how far up. Three dimensions.

But the paper is still *just a sheet of paper*. If you flattened it back out, you'd
only need **2** numbers: how far across, how far down.

- The paper **lives in** 3 dimensions. That's the *ambient* dimension.
- The paper **actually is** 2-dimensional. That's the **intrinsic dimension**.

The crumpling added no new information. It just moved a 2D thing through 3D space.

### Why anyone cares

Real datasets are crumpled paper.

A photo might have 1,000,000 pixels — a million numbers. But photos of faces don't use
that space randomly. They vary in a much smaller number of ways: face angle, lighting,
age, expression, and so on. Maybe a few dozen real degrees of freedom, wearing a
million-number costume.

That "few dozen" is the intrinsic dimension, and it's genuinely useful to know:

- **It tells you how hard your problem is.** Low intrinsic dimension is why machine
  learning works at all on data that looks impossibly high-dimensional.
- **It tells you something about a neural network's insides.** If you look at the
  activations inside a trained network layer by layer, the intrinsic dimension
  changes. Researchers use that pattern to argue about what the network is doing —
  where it's compressing, where it's expanding.

That second use is what this project is about.

---

## Part 2 — How do you measure it?

You can't just count. Nobody hands you the answer. You have to infer it from how the
points are arranged.

### The core intuition: crowding

Here's the trick, and it's genuinely clever.

Imagine you're standing in a crowd. Measure two things:

- `r₁` = distance to the **closest** person
- `r₂` = distance to the **second closest** person

Now take the ratio `r₂ / r₁`. How much farther is the runner-up than the winner?

**In a low-dimensional space** — say everyone is standing in a single-file line — the
second-closest person can easily be *much* farther than the closest. There isn't much
room, so distances spread out. The ratio tends to be large.

**In a high-dimensional space**, there is an enormous amount of room at any given
distance. Space grows *fast* as you step outward. So once you've found your nearest
neighbour, there are tons of other people at almost exactly the same distance. The
runner-up is barely farther than the winner. **The ratio hugs 1.**

That's it. That's the whole idea:

> **The more dimensions there are, the closer the ratio `r₂/r₁` gets to 1.**

So you measure that ratio for every point, average it, and work backwards to the
dimension. The maths makes this exact. On average:

| True dimension | Average ratio `r₂/r₁` |
|---|---|
| 2 | 2.00 |
| 5 | 1.25 |
| 10 | 1.11 |
| 20 | 1.05 |

The tool that does this is called **TwoNN** ("two nearest neighbours"). It's the main
tool the study will test. A closely related tool called **Levina–Bickel** does the same
thing but looks at more neighbours than two.

### The catch, stated plainly

For that table to be true, the maths requires an assumption:

> **Every data point was collected independently of every other one.**

Like drawing marbles from a bag and shaking the bag between each draw. No point knows
anything about any other point.

Hold onto that. It's where everything goes wrong.

---

## Part 3 — What's a confidence interval, and what is "coverage"?

This is the part most people skim, and it's the heart of the project. Slow down here.

### The interval

When a tool estimates something, a good tool also tells you **how sure it is**. It
does that with a confidence interval — a range instead of a single number.

Instead of *"the dimension is 10.3"*, you get *"the dimension is 10.3, and I'm 95%
confident the truth is between 9.6 and 11.2."*

### The promise

That "95%" is a **specific, checkable promise**. It means:

> If you repeated this whole experiment 100 times, roughly 95 of your intervals would
> contain the true answer, and about 5 would miss.

Not "I'm 95% sure." An actual frequency claim about repeated experiments.

### Coverage = auditing the promise

**Coverage** is just: *did the promise hold?* You run the experiment 1,000 times,
count how often the interval actually caught the truth, and compare.

- Caught it 950 times out of 1,000? Coverage is 95%. The tool was honest.
- Caught it 740 times? Coverage is 74%. **The tool was overconfident.** It handed you
  a range that was too narrow and told you to trust it.

### Why this is the thing we care about most

An overconfident interval is *worse than no interval*. A researcher looking at "10.3,
±0.7" concludes the answer is definitely around 10. If the honest range was actually
±3, they've just built an argument on sand — and the number itself looked fine.

Notice that this can happen **even when the estimate is perfectly accurate on
average**. Accuracy and honesty about uncertainty are two different things. That's
exactly the situation the study is designed to detect, which is why coverage is the
primary outcome rather than accuracy.

### The one wrinkle you need to know

To audit a promise, you need to know the true answer. That's why the study will use
synthetic data with a chosen dimension. You can't check a ruler without something of
known length.

---

## Part 4 — What is "clustered sampling"?

Now the other half.

### The households problem

You want to know the average height of adults in a city.

**Approach A:** pick 100 random adults from the phone book. Measure each one. You have
100 independent measurements.

**Approach B:** pick 20 households and measure everyone in each. Also 100 people.

Both give you 100 numbers. **They are not worth the same.**

Family members resemble each other — genetics, nutrition, shared everything. Once
you've measured one person in a household, the next person from that household tells
you *less new information* than a stranger would.

So your 100 measurements from 20 households carry maybe the information of 40
independent people. Maybe 25. It depends on how alike families are.

### Effective sample size

Statisticians call that reduced number your **effective sample size**. The rule:

> Clustered data gives you fewer independent facts than you have rows.

And the consequence that matters here:

> **If you don't account for it, your error bars come out too narrow.**

Because error bars are computed from "how many data points do I have?" — and if your
code counts rows, it thinks you have 100 when you really have 40. It hands you an
interval that's too confident. It made a 95% promise it can't keep.

This is not exotic or controversial. It's textbook survey statistics, and there are
standard formulas for it in that setting.

---

## Part 5 — Where the two halves collide

Here's how researchers actually measure intrinsic dimension inside a language model:

1. Take some sentences.
2. Run them through the model.
3. Grab the internal activation vectors — **for many tokens from each sentence**.
4. Feed that pile of vectors to TwoNN.
5. Report: "layer 12 has intrinsic dimension 15."

Step 3 is the households problem, exactly.

Tokens from the same sentence are **not** independent. They share a topic, a style, a
grammatical context, a speaker. They resemble each other the way family members do.

So we have:

- A tool (Part 2) that **assumes independence**.
- An interval (Part 3) whose **95% promise assumes independence**.
- A collection method (Part 4) that **systematically violates independence**.
- And a literature that reports these numbers without saying how the tokens were
  sampled.

**We searched for prior work measuring what that costs — specifically, the effect on
these estimators and their error bars at realistic sample sizes — and didn't find it.**
That's the gap we're aiming at.

A note on how that's phrased, because it matters: we're saying *we looked and didn't
find it*, not *it doesn't exist*. Those are different claims, and only the first one is
something we can actually back up. One overlooked citation would demolish the second.
Throughout this repo, novelty is stated the careful way — it costs nothing and can't
be embarrassed.

---

## Part 6 — What the study will do

Four steps.

### Step 1 — Build a fake world where we know the answer

The study will generate synthetic data on a shape whose dimension is chosen in
advance, so the right answer is known and the tool can be graded.

It will generate the data in two ways:

- **Independent** — every point drawn on its own. This is our control.
- **Clustered** — points come in groups, and points in a group resemble each other.

A dial called **rho** (ρ) controls how much they resemble each other. `ρ = 0` means
not at all (identical to the control). `ρ = 0.9` means a lot.

### The critical control — read this bit twice

Here's the design choice that makes the whole thing work.

The construction is arranged so that **any single point, looked at on its own, comes
from exactly the same distribution no matter what ρ is.** Same shape, same spread,
same everything — for one point in isolation.

The way that works is almost embarrassingly simple. Each point is a blend of two
ingredients: a **shared** ingredient that everyone in the group gets, and a
**personal** ingredient that's unique to that point. The construction will mix them
at weights that always add up to 1. So the total "amount of spread" never changes —
the mix only shifts how much comes from the group versus the individual.

**But what definitely does change is how points relate to each other.** Turn ρ up, and
groupmates sit closer together. That's not a bug in the experiment. **That's the whole
mechanism.** Because groupmates are closer, they're more likely to be each other's
nearest neighbours, and at shorter distances — and nearest-neighbour distances are
precisely what the tool reads.

So the honest one-liner is:

> **Same one-point distribution. Different relationships between points.**

Not "the same data, just labelled differently." That would be false, and it would make
the whole thing seem impossible — if the data were truly identical, how could the
answer change?

### Step 2 — Grade the tools with the right outcomes

The study will run four estimators, but not every outcome applies to every estimator.

| Planned outcome | Where it applies |
|---|---|
| **Bias, variance, RMSE, and failure rate** | The three nearest-neighbour estimators |
| **Participation-ratio bias and RMSE against the true dimension** | The isotropic linear Gaussian case only |
| **Participation ratio on curved shapes** | Change from its matched `rho = 0, m = 1` baseline, not error against the true dimension |
| **Coverage** | The likelihood-based `TwoNN_MLE` only; both its analytic and planned jackknife intervals |

### Step 3 — Ask how much information was really there

The payoff question:

> Clustered data with 2,500 points behaves like independent data with **how many**
> points?

The study will answer this by measurement. First it will chart how each tool behaves
on *independent* data as it receives more and more points. Then it will take a
clustered result and look up where it lands on that chart. If clustered-2,500 performs
like independent-600, then the implied effective sample size will be about 600.

There's a classic formula from survey statistics that predicts this number for simple
averages. The study will draw it as a **reference line** on the one experiment where
it's legitimate, to see whether these more complicated tools follow it or wander off.
Either answer will be informative.

### Step 4 — Try it on a real model

Finally, the plan includes a small language model on a laptop CPU. The study will hold
the **total number of activation vectors fixed** and change only how they are
collected:

- 4,000 sentences, 1 token each
- 400 sentences, 10 tokens each
- 80 sentences, 50 tokens each
- 16 sentences, 250 tokens each

The number of vectors will stay fixed. The study will then report how much the answer
moves.

**The study will not claim to know the true intrinsic dimension of a real neural
network.** This quantity is not known for the planned real-data example. The step will
show only how much the reported number depends on an arbitrary choice that papers
usually do not mention.

---

## Part 7 — What we expect, and why

These predictions concern the nearest-neighbour estimators; participation ratio is a
separate mechanistic contrast with embedding-specific scoring. In order of confidence:

**1. Coverage gets worse as clustering increases.** *(Most confident.)*
Clustering removes *information* without removing *rows*. Anything that counts rows to
decide how confident to be will be overconfident. This is close to arithmetic.

**2. Variance goes up.** Fewer effective independent points means a noisier estimate.

**3. Bias might barely move.** Genuinely uncertain here — and if that's how it lands,
it's the most interesting outcome, not a boring one. "The number looks fine but the
error bars are lying" is a much more dangerous failure mode than an obviously wrong
number, because nothing looks wrong.

### One honest complication

The interval the study will test is exact only under an *idealised* version of the
maths. In the real geometry, neighbouring points share neighbours, which introduces a
small amount of dependence **even in the clean control condition**. The people who
invented the interval already knew and said this.

So the interval might come in a bit under 95% even with no clustering at all — maybe
91%. **That part isn't our finding, and claiming it would be taking credit for someone
else's known result.**

The study will report **two numbers**, not one, because they answer different
questions:

> **The change from our own control.** If coverage is 91% with no clustering and 74%
> with heavy clustering, **the clustering cost is the drop from 91 to 74.** That's the
> part that's ours, and it's what lets us say clustering *caused* it.

> **The plain number.** 74% is 74%. If a tool promises 95% and delivers 74%, you
> shouldn't use it — and a reader deserves to see that directly rather than having to
> work it out from a difference.

Give only the first and you've hidden whether the tool actually works. Give only the
second and you've claimed someone else's finding as your own. So: both, every time.

We also plan a separate check on perfectly idealised data, where the interval *should*
hit 95% exactly, to prove the interval calculation isn't just buggy.

Three different things could make coverage look bad — a bug, the shared-neighbour
wrinkle, and clustering. The plan will isolate and check each separately. That
separation is most of what makes a small result trustworthy.

---

## Part 8 — What would make us stop

Written down in advance, on purpose. It's much too easy to keep digging until you find
something, and results found that way don't replicate.

- **The Day 1 pilot is an engineering and signal check, not a stopping test.** It will
  check correctness, runtime, effect direction, and whether the Stage A grid needs
  revision. A weak or noisy pilot cannot support a formal null conclusion and cannot
  stop the project merely because its signal looks small.
- **Only the full Stage A experiment can trigger a null-result stop.** That decision
  will use the preregistered thresholds and require the absence of a coherent,
  practically meaningful coverage, bias, or variance effect in the primary decision
  panel.
- **If an analysis turns out to be ill-posed**, the study will fall back to the raw
  measurements rather than forcing an interpretation.
- **If time runs out on the more involved statistics**, the study will ship the simple
  version and put the rest in "future work."
- **If we have a great new idea**, it goes in a parking-lot list. It does not get
  started. That's how a one-week project becomes a three-week project.

All of this lives in [ROADMAP.md](ROADMAP.md) with the exact triggers.

---

## Part 9 — What we are NOT saying

Important enough to repeat:

- ❌ We are **not** saying these tools are broken. They're well-founded when their
  assumptions hold. The study will measure what happens outside them.
- ❌ We are **not** claiming to know the true intrinsic dimension of any real neural
  network.
- ❌ We are **not** claiming clustered sampling is mathematically strange. It's ordinary
  and well-understood in theory. What's missing is a measurement of what it costs
  *these particular tools at realistic sample sizes*.
- ❌ We are **not** saying anyone did anything wrong. We searched and found no prior
  work directly measuring this specific finite-sample estimator and interval behavior
  under externally imposed clustered sampling.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Ambient dimension** | How many numbers you're storing. The crumpled paper's 3. |
| **Intrinsic dimension (ID)** | How many numbers you'd *really* need. The paper's 2. |
| **Manifold** | The mathematical word for the "sheet" — a smooth low-dimensional surface bent through a bigger space. |
| **Nearest neighbour** | The closest other point to the one you're standing on. |
| **TwoNN** | Tool that estimates ID from the ratio of your two closest neighbours' distances. Confusingly, "TwoNN" names **two different tools** — one fits a line to the data, one uses a formula. The study will test both separately, because people often treat them as interchangeable and they aren't. |
| **Levina–Bickel** | Same idea, uses more than two neighbours. |
| **Participation ratio** | A different kind of measure entirely, based on overall spread rather than neighbours. Planned as a contrast, with scoring that depends on the embedding. |
| **i.i.d.** | "Independent and identically distributed." The marbles-from-a-shaken-bag assumption. |
| **Clustered sampling** | Collecting data in groups, where group members resemble each other. The households problem. |
| **rho (ρ)** | Our dial for how much group members resemble each other. 0 = not at all, 0.9 = a lot. |
| **Effective sample size** | How many *independent* data points your clustered data is really worth. |
| **Confidence interval** | A range plus a promise about how often that range contains the truth. |
| **Coverage** | Auditing the promise. How often the range actually contained the truth. |
| **Bias** | How wrong the estimate is on average. |
| **Variance** | How much the estimate bounces around between repeats. |
| **Seed** | The number that makes a random experiment repeatable. |
| **Design effect** | The survey-statistics formula for how much clustering shrinks your effective sample size. |

---

## What to read, in what order

The "Related work" section in the README lists the papers that **bound the planned
contribution** — the ones used to check that the study is not duplicating someone.
Those are chosen for defending novelty, and they are *not* the easiest way to learn
this material.

If you want to actually understand the field, read these instead, in this order:

**1. Start here — the tool itself**
[Estimating the intrinsic dimension of datasets by a minimal neighborhood information](https://www.nature.com/articles/s41598-017-11873-y)
— Facco, d'Errico, Rodriguez & Laio (2017). The original TwoNN paper. Short, published
in a general-audience journal, and the core idea is exactly the `r₂/r₁` ratio
explained in Part 2. This is the estimator the project plans to stress-test.

**2. Why anyone does this to neural networks**
[Intrinsic dimension of data representations in deep neural networks](https://arxiv.org/abs/1905.12784)
— Ansuini, Laio, Macke & Zoccolan (NeurIPS 2019). The paper that kicked off measuring
ID inside networks. They find ID rises through early layers then falls, and that the
last hidden layer's ID predicts test accuracy. Readable, and it's the *motivating*
work — the planned stress test targets the methodology this paper popularised.

**3. The state of the field, if you want breadth**
[A Survey of Dimension Estimation Methods](https://arxiv.org/abs/2507.13887)
— Binnie, Dłotko, Harvey, Malinowski & Yim (2025). A 45-page survey covering the whole
zoo of estimators, organised by what geometric information each one uses, with a
comparison of how they respond to curvature and noise. Its opening motivation is
almost exactly ours: lots of estimators exist, and there's little guidance on using
them reliably. Skim the taxonomy, read the evaluation.

**4. A recent adjacent estimator**
[A Universal Nearest-Neighbor Estimator for Intrinsic Dimensionality](https://arxiv.org/abs/2603.10493)
— Ong, Bobrowski, Reinert & Skraba (2026). It introduces L2N2, another estimator based
on nearest-neighbour distance ratios, with a theoretical universality result across
generating distributions. It belongs to the same broad estimator family, but this
project does not rely on it for claims about clustered dependence or interval coverage.
Evaluating L2N2 under clustering is Future work, not part of the current grid.

**5. The other estimator in the current plan**
Levina & Bickel, *Maximum Likelihood Estimation of Intrinsic Dimension* (NeurIPS 2004).
Short and foundational. Read it after Facco — same family of idea, more neighbours.

**6. For the clustered-sampling half**
You don't need a paper for this one; you need the concept of the **design effect** and
**intraclass correlation** from survey statistics. Any introductory survey-sampling
chapter covers it, and the Part 4 households story above is the whole intuition. The
original source is Leslie Kish's *Survey Sampling* (1965), but a modern textbook
chapter is a gentler entry point.

**Then**, once the above makes sense, the papers in the README's Related work section
will read easily — and you'll see precisely how they bound or sit next to the planned
contribution.
