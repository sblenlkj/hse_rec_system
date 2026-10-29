# Task 03 — ALS + validation diagnostic — Completion Report

## Task

Source: `.artifacts/tasks/003-als-validation-diagnostic.md`

Implemented the scoped ALS comparison on the unchanged chronological 90/10 protocol and a separate deterministic masked-10% diagnostic, then produced the next numbered submission candidate.

## Summary

The new `recsys-als-diagnostic` workflow evaluates Popularity, cosine ItemKNN, binary ALS, and rating-weighted ALS on both protocols using train-only fitting and seen filtering. Rating-weighted ALS is the new local best at 0.067046 chronological Recall@10, 0.009730 above the Task 02 hybrid. Both protocols produce the same model ordering. The best ALS was refit on all events and written as submission candidate №3.

## Files changed

- `src/recsys/split.py` — deterministic per-user masked split and user-item disjointness assertion; the existing chronological split is unchanged.
- `src/recsys/als_models.py` — thin `implicit` ALS adapter supporting binary and raw-rating confidence, deterministic CPU fitting, seen filtering, and popularity fallback.
- `src/recsys/als_pipeline.py` — reproducible two-protocol evaluation, versioned JSON result, model comparison, full-data refit, and submission generation.
- `src/recsys/baselines.py` — accepts the shared optional `n_users` fit keyword while preserving the existing positional `n_items` argument.
- `tests/test_split.py`, `tests/test_als_models.py` — masked reproducibility/disjointness and ALS recommendation invariant coverage.
- `pyproject.toml` — adds the `recsys-als-diagnostic` entry point.
- `notes/hypotheses.md` — records E05/E06 hypotheses, measured results, protocol interpretation, and next step.
- `.artifacts/003_als_validation_diagnostic.json` — machine-readable metrics, hyperparameters, runtimes, rankings, and submission metadata.
- `.artifacts/submissions/3_submission_als.csv` — Kaggle-ready full-data recommendations from rating-weighted ALS.

## Implementation notes

- Masked sampling uses seed 42 independently per user and holds out approximately 10% of distinct user-item pairs, with at least one held-out pair for histories containing at least two distinct items.
- Both ALS variants use `implicit==0.7.3`, 64 factors, regularization 0.01, 15 iterations, one CPU thread, and random state 42. Binary confidence is 1.0; weighted confidence is the raw rating from 1 to 5.
- Validation data is used only as ground truth. Full events are introduced only after local model selection for final refitting.
- The output is numbered №3 because Task 02 already created submission №2, despite the older Task 03 title referring to a second submission.
- The optional blend was intentionally deferred: ALS already answered the task hypothesis with a material gain, so no additional model experiment was needed.
- A concurrent pre-existing change in `notes/plan.md` was preserved and is not part of this task.

## Tests and validation

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` — 11 tests passed.
- `PYTHONPATH=src .venv/bin/python -m compileall -q src tests` — passed.
- `PYTHONPATH=src .venv/bin/python -m recsys.als_pipeline --help` — passed.
- `UV_CACHE_DIR=/tmp/recsys-uv-cache uv lock --check` — passed; 7 packages resolved.
- Full ALS diagnostic command — passed twice on 894,149 events; all Recall@10 values reproduced exactly.
- Independent submission validation — passed for 6,040 users in sample order, exactly 10 unique valid item IDs per row, and no overlap with each user's full observed history.
- `git diff --check` — passed.

## Acceptance criteria

- Chronological protocol unchanged — satisfied; the prior ItemKNN score reproduced exactly at 0.0558165288.
- Separate deterministic masked-10% diagnostic — satisfied and tested for order-independent reproducibility and pair disjointness.
- Binary and rating-weighted ALS — satisfied with two deliberately scoped configurations.
- Chronological and masked Recall@10 plus runtime table — satisfied in durable documentation and the JSON artifact.
- Model ranking comparison — satisfied; both protocols rank weighted ALS, binary ALS, ItemKNN, then Popularity.
- Next Kaggle submission candidate — satisfied at `.artifacts/submissions/3_submission_als.csv`; no automatic upload was performed.
- Synthetic tests — satisfied for the requested split and recommendation invariants.
- Experiment documentation and completion report — satisfied.

## Benchmarks

| Model | Chronological Recall@10 | Masked Recall@10 | Chronological runtime | Masked runtime |
|---|---:|---:|---:|---:|
| Popularity | 0.036744 | 0.069072 | 3.49 s | 3.48 s |
| ItemKNN cosine | 0.055817 | 0.109801 | 4.40 s | 4.45 s |
| ALS binary | 0.064361 | 0.156817 | 2.02 s | 2.03 s |
| ALS rating-weighted | **0.067046** | **0.169748** | 2.77 s | 2.81 s |

The first complete measured run, including full-data refit and submission writing, took 30.26 seconds. A second run took 31.03 seconds and reproduced every score exactly.

## Unresolved items

- Kaggle Recall@10 for submission №2 and submission №3 remains unknown because neither file was uploaded automatically.
- The masked protocol yields much higher absolute scores, but there is not yet evidence that it matches Kaggle hidden-set construction more closely than chronological validation.

## Suggested follow-ups

- Upload `.artifacts/submissions/3_submission_als.csv` manually and record its public Recall@10 before deciding whether masked validation adds predictive value.
- Then implement the required SASRec/BERT4Rec baseline on the unchanged chronological protocol and compare it with the new ALS reference.

## Proposed git commit message

`Add ALS validation diagnostic and third submission candidate`
