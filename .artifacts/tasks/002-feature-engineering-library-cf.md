# Task 02 — Feature engineering + library-based CF

Source request: `.agents/tasks/02_feature_engineering_library_cf.md`.

Build a small reusable train-only feature layer with user/item rating aggregates and user genre profiles. Evaluate a mature-library ItemKNN implementation, a genre/content recommender, and a simple ItemKNN + genre hybrid on the existing chronological 90/10 per-user split. Preserve the custom ItemKNN as the reference, add an explicit leakage sanity check, save versioned machine-readable results, update the experiment journal, and create a new numbered Kaggle submission only if the new model materially improves the 0.055817 reference Recall@10.
