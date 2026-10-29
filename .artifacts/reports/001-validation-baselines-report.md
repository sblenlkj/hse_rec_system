# Task 01 — Leakage-safe validation and baselines — Completion Report

## Task

Source: `.artifacts/tasks/001-validation-baselines.md`

Implemented the first reproducible end-to-end recommendation baseline requested by `.agents/tasks/01_validation_baselines.md`: leakage-safe validation, two popularity baselines, a personalized ItemKNN model, measured Recall@10, and a full-data submission.

## Summary

The repository now exposes `uv run recsys-baseline`, which validates the event schema, builds the fixed chronological 90/10 per-user split, evaluates two popularity variants and cosine ItemKNN, refits ItemKNN on all events, and writes a validated submission plus machine-readable results under `.artifacts/`. ItemKNN reached 0.055817 local Recall@10 versus 0.036744 for global popularity.

## Files changed

- `pyproject.toml`, `uv.lock` — add the minimal NumPy/SciPy runtime and the one-command entry point.
- `src/recsys/split.py` — chronological per-user split and temporal/user-item leakage assertions.
- `src/recsys/metrics.py` — macro Recall@10 with explicit empty-ground-truth and duplicate behavior.
- `src/recsys/baselines.py` — global and rating-threshold popularity with train-side seen filtering.
- `src/recsys/models.py` — sparse cosine ItemKNN with a bounded top-200 neighbor graph.
- `src/recsys/submission.py`, `src/recsys/pipeline.py` — data checks, evaluation orchestration, full-data refit, and strict submission writing.
- `tests/` — synthetic split, metric, duplicate, and seen-filtering tests.
- `notes/hypotheses.md` — measured experiment scores, runtimes, leakage checks, and conclusions.
- `.artifacts/submission_itemknn.csv`, `.artifacts/validation_results.json` — generated review outputs (gitignored).

## Implementation notes

- Validation uses `floor(0.9 * user_history_length)` after sorting each user by timestamp; every dataset user appears in both sides.
- All local popularity counts, seen sets, co-occurrence values, cosine similarities, and candidate scores use only the train partition.
- ItemKNN uses binary feedback. Each item retains its 200 strongest cosine neighbors; recommendations exclude the user's train items.
- Final recommendations are produced only after local evaluation by fitting a fresh ItemKNN instance to all 894,149 events. All observed items are then excluded per user.
- No ensemble or sequential model was added because the task required only one reliable personalized baseline and explicitly prioritized a minimal first solution.

## Tests and validation

- `uv run python -m unittest discover -s tests -v` — 4 tests passed.
- `uv run python -m compileall -q src tests` — passed.
- `uv run recsys-baseline` — passed on 894,149 events; generated scores, results JSON, and submission in 16.95 seconds total.
- Independent CSV validation — passed for 6,040 rows in sample order, exactly 10 unique item IDs per row, and no recommendation intersecting that user's full observed history.
- Data sanity checks — passed for the exact header, non-negative IDs/timestamps, ratings in 1..5, unique user-item pairs, and unique timestamps within each user.

## Acceptance criteria

- Leakage-safe split implemented and checked — satisfied.
- Macro Recall@10 implemented with a synthetic test — satisfied.
- Two popularity baselines evaluated — satisfied: 0.036744 global; 0.036252 rating >= 4.
- At least one personalized model evaluated — satisfied: cosine ItemKNN at 0.055817.
- Local scores and runtimes saved — satisfied in `.artifacts/validation_results.json` and `notes/hypotheses.md`.
- One-command submission generation — satisfied with `uv run recsys-baseline`.
- Reproducible dependencies and deterministic ranking — satisfied through `uv.lock` and stable item-ID tie-breaking.
- Final submission format — satisfied and independently validated.

## Benchmarks

Final full run on the local machine:

| Model/stage | Recall@10 | Runtime |
|---|---:|---:|
| Global popularity | 0.036744 | 3.52 s |
| Popularity, rating >= 4 | 0.036252 | 3.51 s |
| Cosine ItemKNN, 200 neighbors | 0.055817 | 4.46 s |
| Full-data ItemKNN refit + submission | n/a | 4.67 s |
| Complete command | n/a | 16.95 s |

## Unresolved items

- Kaggle Recall@10 was not measured because no submission upload was requested or performed.
- The local score remains below the cited public leaderboard baseline; this task establishes the honest working ladder rather than a competitive final model.

## Suggested follow-ups

- Evaluate binary and rating-weighted ALS on the unchanged split.
- If ALS adds complementary hits, test one deterministic ALS + ItemKNN rank blend before moving to SASRec.

## Proposed git commit message

`Add leakage-safe validation and ItemKNN baseline pipeline`
