# Task 04 — SASRec via RePlay, local benchmark + Colab-ready training — Completion Report

## Task

Source: `.artifacts/tasks/004-sasrec-colab.md`

Implemented one leakage-safe RePlay SASRec baseline on the existing chronological 90/10 split, including a runtime-based local/Colab decision, full local validation, reusable checkpoints and top-200 candidates, full-data refit, submission #4, and a Run All Colab workflow.

## Summary

RePlay SASRec 0.22.0 was integrated through its current block-based API. The full CPU experiment produced local Recall@10 **0.130385**, improving the rating-weighted ALS reference 0.067046 by **0.063339 absolute**. Both validation and full-data top-200 candidates contain 1,208,000 rows (200 for each of 6,040 users), and the final submission contains ten unique full-history-unseen items per user.

## Files changed

- `src/recsys/sasrec_data.py` — deterministic train-fitted item encoding, ordered user sequences, and leakage assertions.
- `src/recsys/sasrec_pipeline.py` — RePlay training, validation, checkpoint reload, candidates, full refit, submission, metrics, and zip bundle orchestration.
- `tests/test_sasrec_data.py` — sequence ordering, train-only vocabulary, unknown-item encoding, and leakage failure coverage.
- `notebooks/04_sasrec_colab.ipynb` — clean-runtime GPU check, pinned project install, raw CSV upload, full pipeline execution, artifact validation, and bundle download.
- `pyproject.toml`, `uv.lock` — pinned `replay-rec[torch]==0.22.0` optional dependency and `recsys-sasrec` entry point.
- `notes/hypotheses.md` — E07 configuration, measured result, runtime decision, artifacts, and Task 05 handoff.

## Implementation notes

- Local validation fits the catalog, sequences, model, candidates, and seen filter only from chronological train. Full events are used only after local scoring for the final refit.
- Ratings are intentionally not SASRec inputs; the model consumes ordered item IDs. This keeps the first transformer experiment minimal and comparable without a hyperparameter sweep.
- The current RePlay `ParquetModule` / `SasRec.from_params` / Lightning API is used because the former legacy pipeline has changed.
- The smoke estimate was far below the 45-minute cutoff, so the full experiment ran locally on CPU. MPS was built but unavailable in this runtime. The Colab workflow remains available for repeat GPU runs.
- Existing Task 03 changes and the unrelated user edit in `notes/plan.md` were preserved.

## Tests and validation

- `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` — 13 tests passed.
- `PYTHONPATH=src ... python -m recsys.sasrec_pipeline --mode smoke --device cpu` — passed; preprocessing, training loop, checkpoint reload, and recommendations validated.
- `PYTHONPATH=src ... python -m recsys.sasrec_pipeline --mode full --device cpu` — passed; validation, top-200 generation, full refit, checkpoint, submission, JSON, and zip created.
- Independent pandas artifact audit — passed: 200 candidates/user, unique user-item pairs, ranks 1–200, finite scores; submission order and cardinality valid with zero full-history seen overlaps; zip CRC passed.
- Full checkpoint reload plus one-batch inference — passed for 128 users / 1,280 recommendations.
- Notebook JSON parse and Python AST compilation of all code cells — passed (11 cells).
- `UV_CACHE_DIR=/tmp/recsys-uv-cache uv lock --check` — passed.
- Actual Google Colab Run All — not run because the measured local path met the runtime criterion and completed successfully; GPU-specific execution remains unmeasured.

## Acceptance criteria

- **Satisfied:** ready library implementation; local smoke; measured local/Colab decision; unchanged chronological validation; local Recall@10; ALS comparison; reloadable checkpoints; scored/ranked top-200 candidates; versioned JSON; numbered submission; hypothesis documentation.
- **Satisfied with static validation:** Run All Colab notebook and deterministic raw-CSV upload/download workflow. The notebook parses and its Python cells compile, but a Colab GPU session was not executed.
- **Satisfied:** output is directly usable by Task 05 through `user_id`, `item_id`, `sasrec_score`, and `sasrec_rank`.

## Benchmarks

- Smoke: 8 train + 4 validation batches on CPU, 2.24 s; estimated 13.42 s/epoch and 134.20 s for ten validation epochs.
- Full validation training: 153.87 s on CPU; best epoch 9 (zero-based); Recall@10 0.130385.
- Full-data refit: 10 epochs, 102.41 s on CPU.
- Complete full pipeline: 277.27 s.
- Colab GPU: not used and therefore no runtime or GPU type is reported.

## Unresolved items

- Kaggle Recall@10 for `.artifacts/submissions/4_submission_sasrec.csv` is not yet known.
- The Colab notebook has not been executed in an actual Colab GPU runtime.

## Suggested follow-ups

- Submit submission #4 and record its Kaggle score before drawing leaderboard conclusions from the large local uplift.
- Use the saved SASRec top-200 alongside ALS, ItemKNN, and popularity candidates in Task 05 two-stage reranking.

## Proposed git commit message

`Add reproducible RePlay SASRec baseline and Colab workflow`
