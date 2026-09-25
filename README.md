# AI-Validation

This repository contains the `aiv` Python package, the companion notebooks for the book
*AI for Validation* and the synthetic anti-money-laundering (AML) evidence that the
notebooks read. The book manuscript is not included. Every notebook runs from a pip
installation without a checkout of the authors' working repository.

The case study throughout is a frozen logistic model of a rare AML outcome. The notebooks
locate customer regions where the model's Brier loss is elevated, search over those
regions with and without a language-model structural proposal, confirm the frozen
selections on 200,000 new observations and check reported values against a source-bound
evidence store.

## Installation

Python 3.10 or later is required.

```bash
pip install "aiv[notebooks] @ git+https://github.com/asudjianto-xml/AI-Validation.git"
```

or, from a clone of this repository:

```bash
pip install ".[notebooks]"
```

The `notebooks` extra adds Matplotlib, joblib, ipykernel and scikit-learn-extra.
The Chapter 2 notebook uses scikit-learn-extra's `KMedoids` to replay a recorded PAM
clustering. Its compiled extension requires NumPy 1.x, so the extra pins `numpy<2`, and
it imports `distutils`, so the extra installs setuptools on Python 3.12 and later.
Prebuilt scikit-learn-extra wheels exist for Python 3.6 to 3.11 on x86-64; on other
platforms pip compiles it from source, which needs a C compiler. XGBoost 3 on Linux also
installs the `nvidia-nccl-cu13` wheel, a download of about 290 MB.

The saved model artifacts were produced with scikit-learn 1.6.1 and XGBoost 3.4.0. The
notebooks have also been executed with scikit-learn 1.9.1 and XGBoost 3.4.1, and each
notebook's `execution_summary.json` records the versions used for its saved outputs.

Two further extras are optional: `kg` installs `knowlytix` for `aiv.kg_rag`, and `torch`
installs PyTorch for the space-filling design optimizer and `aiv.moegp`.

## Companion notebooks

```bash
aiv-companions copy ./aiv-notebooks   # copy the notebooks to a writable directory
aiv-companions where                  # print where the evidence data was found
jupyter lab ./aiv-notebooks
```

The copied notebooks may be opened from any directory. Each setup cell locates the
evidence data in this order: the directory named by the `MODEL_VALIDATION_ROOT`
environment variable, a repository checkout containing the current directory, then the
data installed with the package. It then checks the SHA-256 digests that link the
evidence files before any value is used.

`regional_case/` holds one walkthrough per book chapter. Each code cell is preceded by
an explanation of what it computes and followed by an interpretation of its recorded
output.

| Chapter | Notebook | Content |
|---|---|---|
| 1 | `ch01_foundations` | One medoid candidate from search to frontier, frozen selection, confirmation and a source-bound claim check |
| 2 | `ch02_evaluator` | Frozen-model Brier losses, a refitted depth-two error model, weighted leaf coordinates, PAM and confirmation of the selected region |
| 3 | `ch03_search` | Search budgets, candidate lineage, frontier recomputation and the twinning-versus-random confirmation comparison |
| 4 | `ch04_reflection` | A recorded language-model proposal, its admission and its matched-budget comparison with continued search |
| 5 | `ch05_evidence_store` | Rebuilding the evidence store and resolving records to their source fields |
| 6 | `ch06_verify_claims` | Typed claim verification, missing evidence and corrupted-record handling |
| 7 | `ch07_adversarial_prompts` | A crossed design of claim mutations against the verifier |
| 8 | `ch08_findings` | Thirteen confirmed regional findings, selection optimism and paired simultaneous intervals |
| 9 | `ch09_report` | Evidence packets, content-plan checks and a source-checked report section |

`regional_live/` holds nine notebooks that inspect a later end-to-end run with live
language-model calls: a structural proposal, 70 answer tests and nine report drafts.
The notebooks read the recorded calls and make no new ones.

No notebook fits the frozen predictor or calls a language model. Chapter 2 refits the
auxiliary error model in memory and checks it against the saved model.

## Data

The evidence is installed under `aiv/companions/_data/`. It keeps the relative paths of
the authors' working repository, for example
`book/evidence/aml_medoid_loop_leaf_output_20260921/results.json`, because the evidence
records identify their source files by those paths and digests. Its main contents are:

- `book/evidence/aml_error_clustering_current_20260921/`: the frozen predictor, the
  auxiliary error model and the 40,000 discovery observations;
- `book/evidence/aml_leaf_geometry_weighted_20260924/`: the Chapter 2 geometry, its
  selections and 200,000 confirmation observations;
- `book/evidence/aml_medoid_loop_leaf_output_20260921/`: the medoid search histories,
  the recorded structural proposal and admission, frozen finalists and 200,000
  confirmation observations;
- `book/evidence/parallel_aml_20260921/`: the source-bound evidence store;
- `book/experiments/regional_live/runs/20260923T220909Z/`: the live run's records.

The observations are simulated account-months from a synthetic population. The
measurements describe the frozen model in that population and do not establish
operational AML performance.

## The `aiv` package

| Module | Purpose |
|---|---|
| `aiv.weak_region`, `aiv.weak_kernel` | Weak-region discovery over the residual field and over an error model |
| `aiv.wsearch` | Constrained weakness search: representation, evaluator and cost accounting |
| `aiv.frontier_selection`, `aiv.frontier_weakregion` | Frontier ownership, parent selection and frontier-based discovery |
| `aiv.confirmation` | Confirmation of a frozen region on held-out observations |
| `aiv.claim_verify` | Deterministic claim verification against an evidence store |
| `aiv.report`, `aiv.report_instruction` | Report assembly, section scoring and instruction optimization |
| `aiv.predictive` | A predictive adapter and functional-ANOVA model |
| `aiv.monotone_moe`, `aiv.moegp` | Mixture-of-experts models over XGBoost leaf structure |
| `aiv.companions` | Locating and copying the companion notebooks and data |

`aiv.predictive` accepts a CSV path; this distribution does not bundle the credit
example that `aiv.default_data_path()` refers to. `aiv.kg_rag` requires a GMS store, and
`aiv.aml_workflow` reads historical studies from the working repository; neither
resource is included here.

`aiv/_vendor/` contains code copied from Frontier-Discovery so that the package has no
dependency on another repository.

## Tests

```bash
pip install ".[notebooks]" pytest
python -m pytest tests
```

## License

Apache License 2.0; see `LICENSE`.

This repository is exported from the authors' working repository, where changes are
made.
