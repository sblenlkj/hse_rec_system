# Task 05 — обязательный two-stage recommender: candidate union + reranking — Completion Report

## Task

Source: `.artifacts/tasks/005-two-stage-reranking.md`

Implemented the mandatory two-stage recommender with a direct temporal 80/10/10 ranker protocol, SASRec/ALS/ItemKNN/popularity candidate union, train-only features, simple blending baselines, CatBoostRanker, full-data inference, and submission #5.

## Summary

The final candidate union raised Candidate Recall from SASRec@200 **0.598884** to **0.758038**. A frozen CatBoost YetiRank model improved Recall@10 from the same-holdout SASRec score **0.132715** to **0.144796** (+0.012081 absolute); relative to the Task 04 reference 0.130385, the gain is +0.014411. Simple normalized-score and reciprocal-rank blends were both worse than SASRec.

## Files changed

- `src/recsys/split.py` — deterministic direct 80/10/10 per-user split and three-layer leakage assertions.
- `src/recsys/als_models.py` — native ALS score/rank output for candidate union.
- `src/recsys/two_stage.py` — unified source table, stable missing handling, candidate diagnostics, labels, compact features, and deterministic ranking helpers.
- `src/recsys/two_stage_pipeline.py` — end-to-end fixed-epoch generators, train-only feature layers, CatBoost training/evaluation, artifacts, and submission.
- `tests/test_split.py`, `tests/test_als_models.py`, `tests/test_two_stage.py` — protocol, scored adapter, union, sentinel, label, and recall coverage.
- `pyproject.toml`, `uv.lock` — pinned CatBoost 1.2.10 ranking extra and `recsys-two-stage` entry point.
- `notes/hypotheses.md`, `AGENTS.md` — E08 result and updated confirmed SASRec leaderboard reference.

## Implementation notes

- Ranker training generators and features use only 712,916 base-train events. The next 89,100 events provide labels only. No final-validation value is passed to ranker fitting or early stopping.
- For final evaluation, SASRec/ALS/ItemKNN/popularity and all aggregates are rebuilt from the first 802,016 events; the frozen ranker is evaluated on the final 92,133 events.
- SASRec uses a fixed ten epochs in Task 05 to avoid early stopping against either ranker labels or final validation.
- Missing scores are zero after per-user/source min-max normalization; missing ranks use `source_limit + 1`; explicit source flags disambiguate sentinels.
- Ranker training retained the complete candidate union: 2,385,924 rows, 60,179 positive rows, and 45 features. No negative downsampling was used.
- Full inference reuses the Task 04 fixed-ten-epoch full SASRec candidates and refits all classical sources/features on full observed events. This preserves the already validated model output instead of retraining an identical full SASRec configuration.

## Tests and validation

- `PYTHONPATH=src /tmp/hse-recsys-replay-smoke/bin/python -m unittest discover -s tests -v` — 16 tests passed with all ranking extras installed.
- `python -m compileall -q src tests` — passed.
- `UV_CACHE_DIR=/tmp/recsys-uv-cache uv lock --check` — passed.
- `git diff --check` — passed.
- Full `recsys.two_stage_pipeline --device cpu` run — passed on all 894,149 interactions and created every requested artifact.
- Independent parquet audit — passed: 6,040 users, unique user-item rows, valid 1–4 source counts; row counts 2,385,924 / 2,410,825 / 2,461,856 for ranker/final/full tables.
- CatBoost checkpoint reload and 2,000-row inference — passed with finite predictions.
- Independent submission audit — passed: 6,040 users in sample order, exactly ten unique items each, zero overlaps with full observed history.

## Acceptance criteria

- **Satisfied:** deterministic leakage-safe 80/10/10 ranker protocol with sanity checks.
- **Satisfied:** RePlay SASRec top-200, rating-weighted ALS top-200, library ItemKNN top-200, and popularity top-50.
- **Satisfied:** individual and incremental union Candidate Recall diagnostics saved in `.artifacts/005_candidate_recall.json`.
- **Satisfied:** unified one-row-per-pair source table with scores, ranks, flags, normalization, and `n_sources`.
- **Satisfied:** train-only user/item/genre/affinity features with raw demographics.
- **Satisfied:** SASRec, normalized-score blend, and reciprocal-rank blend baselines.
- **Satisfied:** CatBoostRanker 1.2.10 with YetiRank, 300 trees, depth 7, learning rate 0.05, L2 5, seed 42.
- **Satisfied:** learned reranker evaluated against SASRec on an untouched final 10% holdout.
- **Satisfied:** reloadable `.artifacts/models/5_two_stage_ranker.cbm`.
- **Satisfied:** `.artifacts/submissions/5_submission_two_stage.csv` created without automatic Kaggle upload.

## Benchmarks

- SASRec base-80 fixed fit: 106.12 s CPU; SASRec train-90 fixed fit: 105.19 s CPU.
- CatBoostRanker fit: 416.14 s CPU.
- Full pipeline: 727.08 s CPU, including candidates, features, training, inference, parquet artifacts, and submission.
- Final Candidate Recall: SASRec 0.598884; +ALS 0.718612; +ItemKNN 0.756941; +popularity 0.758038.
- Final Recall@10: SASRec 0.132715; normalized score blend 0.126825; reciprocal-rank blend 0.104150; CatBoost 0.144796.

## Unresolved items

- Kaggle Recall@10 for submission #5 is unknown until manual upload.
- The local uplift does not yet establish that ranker feature distributions transfer to the Kaggle hidden set.

## Suggested follow-ups

- Submit `.artifacts/submissions/5_submission_two_stage.csv` and record its score before adding features.
- If leaderboard transfer is weak, test one additional historical ranker cutoff or score calibration to address temporal distribution shift; if transfer is strong, inspect feature importance and try only one targeted refinement.

## Proposed git commit message

`Add leakage-safe two-stage candidate union and CatBoost reranker`
