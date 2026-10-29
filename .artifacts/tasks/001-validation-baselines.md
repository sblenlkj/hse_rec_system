# Task 01 — Leakage-safe validation and baselines

Source request: `.agents/tasks/01_validation_baselines.md`.

Build the first reproducible end-to-end competition solution: inspect the local data, implement a chronological 90/10 per-user split with leakage invariants, implement macro Recall@10, compare global and rating-aware popularity, train at least one personalized ALS or ItemKNN model, measure local scores and runtimes, refit the chosen model on all events, and generate a valid 10-item-per-user submission with seen-item filtering. The complete workflow must run with one command and the experiment log must record factual results.
