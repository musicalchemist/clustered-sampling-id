# Repository context

## What this project does

`clustered-sampling-id` is the plan for a one-week empirical study of how grouped
sampling changes the behavior of intrinsic-dimension estimators. The study will
measure the estimator-specific outcomes defined below and—most importantly—
confidence-interval coverage while preserving the one-point marginal distribution
across dependence conditions.

The motivating use case is sampling many token activations from a small number of
sequences. The project asks whether estimators and intervals that assume independent
observations remain reliable when observations arrive in dependent groups. It does
not claim to recover the true intrinsic dimension of a real neural representation.

The current written design is frozen during implementation. [CLAUDE.md](../CLAUDE.md)
holds durable design facts, known traps, and preregistered Stage A thresholds;
[ROADMAP.md](../ROADMAP.md) is the source of truth for progress, next actions, branch
names, stopping criteria, and Future work.

## Current state

Day 0 planning and scaffolding are complete. The package currently contains only an
empty `src/csid/__init__.py`; `docs/repo-context.md` is the sole file under `docs/`.
Generators, estimators, inference, experiments, tests, result directories, the report,
and figures will be introduced in roadmap order. Do not mistake planned paths in the
documentation for existing implementation.

## Tech stack

- Python 3.12
- `uv` for environment, dependency, lockfile, and command management
- NumPy, SciPy, scikit-learn, scikit-dimension, Numba, pandas, PyArrow, Matplotlib,
  and Joblib for the planned synthetic study
- Pytest for tests
- Optional PyTorch, Transformers, and Datasets dependencies for the timeboxed
  transformer-protocol arm only
- `uv_build` as the package build backend

Dependencies are declared in `pyproject.toml` and resolved in `uv.lock`. Do not add,
update, or install dependencies without approval.

## Planned architecture and data flow

- `src/csid/`: planned reusable generators, embeddings, estimators, inference,
  jackknife, and metrics code.
- `experiments/`: planned stage-specific runnable scripts. Each experiment will record
  its seed and write raw per-run rows before any plotting.
- `results/raw/`: planned large, reproducible per-run Parquet files. This path will be
  local and gitignored.
- `results/summaries/`: planned small per-condition CSV or Parquet aggregates. These
  will be committed and must be sufficient to reproduce reported numbers and figures.
- `docs/figures/`: planned committed figures built only from committed summaries,
  with the producing Git SHA recorded.
- `docs/REPORT.md`: the planned final report.

The intended flow is:

```text
seeded condition -> marginal-preserving generator -> frozen embedding
                 -> one shared nearest-neighbor pass -> estimators and intervals
                 -> raw per-run rows -> committed summaries -> figures/report
```

## Load-bearing design rules

1. Every dependence condition has the same one-point marginal distribution; only the
   joint dependence structure changes.
2. `m = 1` is the i.i.d. control and uses the same generator code path.
3. Fixed-`n` and fixed-`G` comparisons remain separate.
4. Bias, variance, RMSE, and failure rate apply to the three nearest-neighbour
   estimators. Participation-ratio bias/RMSE against true `d` applies only to the
   isotropic linear Gaussian embedding; nonlinear participation ratio uses its matched
   baseline.
5. Analytic and current-scope jackknife interval coverage apply only to `TwoNN_MLE`.
6. Embedding parameters are frozen within comparison slices and use a distinct,
   recorded `embedding_seed`.
7. Seeds are never reduced to make room for a larger grid.

This summary is not a substitute for the exact constructions and caveats in
[CLAUDE.md](../CLAUDE.md).

## Common workflows

```bash
# Create/synchronize the locked project environment
uv sync

# Run the test suite once tests exist
uv run pytest

# Run future project commands inside the managed environment
uv run <cmd>
```

There is no configured lint, format, or type-check command yet. Do not add tooling or
invent commands solely to fill that gap. Work one roadmap day at a time and use the
roadmap's named branch for that day's pull request.

## Configuration and external systems

No environment variables, credentials, database, hosted service, deployment target,
or default external API are currently required.

The optional transformer arm will eventually consume a fixed public corpus and model
checkpoint, but neither source is selected in the current scaffold. The planned
`intRinsic` comparison is a correctness reference, not a runtime service. Any future
source, revision, cache requirement, credential, or network dependency must be
documented when introduced and handled according to [SECURITY.md](../SECURITY.md).

## Key documents

- [README.md](../README.md): public overview, hypothesis, non-claims, and setup.
- [ELI5.md](../ELI5.md): plain-language explanation and literature reading path.
- [CLAUDE.md](../CLAUDE.md): authoritative technical design and traps.
- [ROADMAP.md](../ROADMAP.md): living execution plan and current status.
- [AGENTS.md](../AGENTS.md): agent workflow, workspace, and safety instructions.
- [SECURITY.md](../SECURITY.md): vulnerability reporting and repository security
  baseline.

## Known limitations and open work

- No experiment implementation or empirical results exist yet.
- Measured runtimes are not yet available, so later grids are provisional.
- Raw results will remain local rather than entering the public repository; planned
  committed summaries and figures will provide the reproducible public artifact.
- New experiment ideas belong in ROADMAP.md Future work and must not expand the
  current project scope.
