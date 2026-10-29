# Task 03 — ALS + validation diagnostic + следующий submission

Source request: `.agents/tasks/03_als_validation_diagnostic.md`.

## Scope

- Preserve chronological per-user 90/10 as the primary leakage-safe protocol.
- Add a deterministic masked-10% diagnostic split with disjoint user-item pairs.
- Evaluate Popularity, cosine ItemKNN, binary ALS, and rating-weighted ALS on both splits.
- Use at most two or three motivated ALS configurations; do not run a broad search.
- Record Recall@10 and runtime, and compare model rankings across protocols.
- Refit the best chronological ALS on all events and create the next numbered Kaggle-ready submission without uploading it.
- Add synthetic tests for masked split reproducibility/disjointness and ALS recommendation invariants.
- Update `notes/hypotheses.md` and create a factual completion report.

## Constraints

- All fitting, candidate generation, and seen filtering must use only the train side of each validation split.
- Do not add transformers, CatBoost, a large refactor, or a broad feature/hyperparameter search.
- A simple ItemKNN + ALS blend is optional and only justified if ALS is not worse.

## Definition of done

The unchanged chronological protocol and new masked protocol both have measured results; ALS binary and preferably rating-weighted are evaluated; runtimes and model rankings are recorded; the next submission candidate exists; tests pass; documentation explains the best ALS, protocol conclusion, submission path, and one next step.
