# Task 02 — Feature engineering + library-based CF — Completion Report

## Task

Source: `.artifacts/tasks/002-feature-engineering-library-cf.md`

Implemented the scoped Task 02 experiment: a train-only feature layer, library cosine ItemKNN, genre content ranking, and a three-alpha KNN + genre hybrid comparison on the unchanged chronological validation protocol.

## Summary

The new `uv run recsys-features-cf` workflow builds leakage-checked user/item/genre features, reruns the custom reference, evaluates `implicit==0.7.3` ItemKNN and genre models, saves versioned metrics, and conditionally refits the best new model on all events. The best hybrid (`alpha=0.25`) reached 0.057316 Recall@10, improving the 0.055817 Task 01 reference by 0.001499 absolute (+2.69%), so submission candidate №2 was created.

## Files changed

- `pyproject.toml`, `uv.lock` — add `implicit` and the `recsys-features-cf` entry point.
- `src/recsys/features.py` — reusable user/item aggregates, two user-genre profiles, source fingerprints, and the exact-train leakage assertion.
- `src/recsys/library_models.py` — adapter for `implicit.nearest_neighbours.CosineRecommender`, genre content ranking, and normalized hybrid reranking.
- `src/recsys/feature_pipeline.py` — one-command evaluation, model selection, versioned result writing, and conditional full-data refit/submission.
- `tests/test_features.py`, `tests/test_library_models.py` — aggregate/profile, leakage rejection, seen filtering, library adapter, and hybrid ordering tests.
- `notes/hypotheses.md` — feature definitions, measured score table, conclusion, submission path, and next experiment.
- `.artifacts/002_feature_engineering_library_cf.json` — machine-readable metrics and runtime record.
- `.artifacts/submissions/2_submission_library_genre_hybrid.csv` — Kaggle-ready full-data recommendations.

## Implementation notes

- User features contain interaction count, mean/std rating, rating >= 4 count/fraction, and rating 1–5 fractions. Item features contain the analogous required aggregates plus all 18 genre indicators.
- The feature layer builds L2-normalized rating-weighted and positive-only genre profiles. Only the positive-only profile was used in this minimal model comparison.
- The library baseline uses binary train feedback, `CosineRecommender(K=200)`, one thread, and 100 candidates per user. No new custom CF algorithm was introduced.
- The hybrid formula is `minmax(knn_score) + alpha * minmax(genre_cosine)` over library ItemKNN candidates for alpha values 0.10, 0.25, and 0.50.
- Feature provenance stores the exact source event count and SHA-256 fingerprint. Local fitting asserts that this fingerprint matches the train partition, not train plus validation.
- The full event set is used only after local model selection, for final feature/model refit and full-history seen filtering.

## Tests and validation

- `uv run python -m unittest discover -s tests -v` — 9 tests passed.
- `uv run python -m compileall -q src tests` — passed.
- `uv run recsys-features-cf --help` — passed.
- `uv lock --check` — passed; 7 packages resolved.
- `uv run recsys-features-cf` — passed on 894,149 events in 10.62 seconds; produced versioned results and submission candidate №2.
- Independent submission validation — passed for 6,040 rows in sample order, exactly 10 unique valid item IDs per row, and zero overlap with each user's full observed history.

## Acceptance criteria

- Mature library used for new CF experiment — satisfied with `implicit==0.7.3`.
- No new handwritten ItemKNN — satisfied; only a thin adapter around `CosineRecommender` was added.
- Reusable leakage-safe feature layer — satisfied and covered by a test that rejects full-event feature provenance.
- Train-only user genre profile — satisfied with two normalized variants.
- Content/genre recommendation evaluated — satisfied at 0.014010 Recall@10.
- Simple KNN + genre hybrid evaluated — satisfied for three alpha values.
- Local Recall@10 comparison table — satisfied in `notes/hypotheses.md` and the machine-readable artifact.
- Submission on material improvement — satisfied; the best hybrid exceeded the configured 0.0005 absolute threshold.
- Experiment documentation updated — satisfied.
- ALS, transformers, boosted ranking, and broad feature search excluded — satisfied.

## Benchmarks

Measured by the completed Task 02 run:

| Model/stage | Recall@10 | Runtime |
|---|---:|---:|
| Feature build + leakage check | n/a | 0.55 s |
| Custom ItemKNN reference rerun | 0.055817 | 4.43 s |
| `implicit` cosine ItemKNN | 0.055680 | 1.30 s |
| Positive genre content model | 0.014010 | 1.28 s |
| Hybrid alpha=0.10 | 0.056435 | 0.09 s |
| Hybrid alpha=0.25 | **0.057316** | 0.09 s |
| Hybrid alpha=0.50 | 0.056007 | 0.09 s |
| Full-data best-model refit + submission | n/a | 2.01 s |
| Complete command | n/a | 10.62 s |

## Unresolved items

- Kaggle Recall@10 for submission №2 is unknown because the file was not uploaded automatically.

## Suggested follow-ups

- Upload `.artifacts/submissions/2_submission_library_genre_hybrid.csv` manually and record its Kaggle score.
- Run the planned ALS diagnostic on the same split and measure whether ALS adds complementary candidates to the new hybrid reference.

## Proposed git commit message

`Add leakage-safe features and implicit ItemKNN genre hybrid`
