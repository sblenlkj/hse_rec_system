# Task 05 — обязательный two-stage recommender: candidate union + reranking

## Цель

Построить полноценную двухстадийную рекомендательную систему, которую преподаватель ожидает увидеть в проекте:

```text
candidate generation
→ candidate union
→ leakage-safe features
→ learning-to-rank / reranking
→ top-10
```

Это обязательный этап проекта.

Текущий best single-stage:

- RePlay SASRec
- local chronological Recall@10 = **0.130385**
- Kaggle Recall@10 = **0.1889**

Two-stage система должна сравниваться именно с SASRec.

---

## 0. Обязательно прочитать

Перед изменениями:

- `AGENTS.md`
- `notes/engineering.md`
- `notes/hypotheses.md`
- reports Task 02–04
- существующий feature layer
- ALS / ItemKNN / SASRec adapters and candidate outputs

Не переписывать существующие модели без необходимости.

---

## 1. Ключевая проблема: не обучать ranker на той же validation, на которой его оцениваем

Нельзя:

```text
first 90% -> base models
last 10% -> labels for CatBoost
same last 10% -> evaluation
```

Это leakage / overfitting validation.

Для two-stage эксперимента нужен отдельный temporal layer.

### Рекомендуемый protocol: 80/10/10 per user

Для каждого пользователя по timestamp:

- первые 80% -> `base_train`
- следующие 10% -> `ranker_train_target`
- последние 10% -> `final_validation`

#### Ranker training data

1. fit candidate generators только на `base_train`;
2. generate top-N candidates для каждого user;
3. label candidate = 1, если item находится в `ranker_train_target`;
4. features строить только из `base_train`.

#### Final validation

1. refit candidate generators на первых 90% истории (`base_train + ranker_train_target`);
2. generate candidates для final 10%;
3. features строить только из первых 90%;
4. применить **уже обученный** ranker;
5. оценить Recall@10 на final 10%.

Таким образом итоговый Recall@10 сопоставим с прежним chronological 90/10 holdout по последним 10% пользователей.

Все split должны быть deterministic и покрыты sanity checks.

---

## 2. Candidate generators

Использовать готовые существующие модели/библиотеки.

Минимум:

### SASRec
- основной источник;
- top-200;
- score;
- rank.

### rating-weighted ALS
- top-100 или top-200;
- score;
- rank.

### ItemKNN
- library-based `implicit` implementation;
- top-100;
- score;
- rank.

### Popularity
- top-50 как fallback/source.

Не писать новые CF алгоритмы.

---

## 3. Сначала измерить Candidate Recall

До ranker обязательно посчитать:

- SASRec Recall@50/100/200;
- ALS Recall@100/200;
- ItemKNN Recall@100/200;
- union Recall@N.

Минимум сравнить:

```text
SASRec
SASRec ∪ ALS
SASRec ∪ ALS ∪ ItemKNN
SASRec ∪ ALS ∪ ItemKNN ∪ Popularity
```

Главный вопрос:

> добавляют ли classical models релевантные items, которых нет в SASRec candidates?

Если union почти не увеличивает Candidate Recall, это важный отрицательный результат и ranker не должен искусственно усложняться.

---

## 4. Candidate table

Для каждой `(user_id, item_id)` пары сохранить одну строку.

Минимальные source features:

- `sasrec_score`
- `sasrec_rank`
- `from_sasrec`
- `als_score`
- `als_rank`
- `from_als`
- `knn_score`
- `knn_rank`
- `from_knn`
- `popularity_rank`
- `from_popularity`
- `n_sources`

Missing source scores/ranks заполнять устойчивым documented способом.

Scores разных моделей нельзя складывать напрямую без normalization.

---

## 5. Leakage-safe ranking features

Переиспользовать feature layer Task 02.

Начать с компактного набора.

### User
- interaction_count
- mean_rating
- std_rating
- fraction_positive
- age
- gender

### Item
- interaction_count / popularity
- mean_rating
- std_rating
- fraction_positive
- 18 genre indicators

### User-item
- positive genre affinity
- rating-weighted genre affinity
- при наличии простой latent similarity от ALS

### Candidate-source
- все score/rank/source features из предыдущего раздела.

Не делать 200–300 features на первой итерации.

Все aggregates для ranker training строятся только по `base_train`.
Все aggregates для final validation строятся только по первых 90%.

---

## 6. Ranker

Использовать готовую библиотеку.

Первый выбор:

- **CatBoostRanker**

Альтернатива:
- LightGBM LGBMRanker, если CatBoost создаёт проблемы.

Не писать ranker вручную.

### Objective

Использовать подходящий ranking objective, например:
- YetiRank / YetiRankPairwise для CatBoost;
- LambdaRank/Lambdarank для LightGBM.

Group = user.

Не делать большой hyperparameter sweep.

Одна разумная конфигурация + максимум 2–3 маленькие вариации.

---

## 7. Negative candidates / imbalance

Для каждого user positive items могут быть редкими среди сотен candidates.

Не использовать все тысячи возможных items.

Работать только с candidate union.

Если candidate pool слишком большой, допускается deterministic negative downsampling, но:
- сохранить все positives;
- сохранить source diversity;
- документировать sampling.

---

## 8. Baselines для reranking

До CatBoost оценить:

1. SASRec top-10 как single-stage baseline.
2. simple normalized score blend SASRec + ALS.
3. simple rank blend SASRec + ALS (+ KNN) — максимум один-два варианта.
4. CatBoost/LightGBM ranker.

Это покажет, нужен ли ML ranker вообще.

---

## 9. Итоговая evaluation table

Минимум:

| Model | Final Recall@10 |
|---|---:|
| SASRec single-stage | ... |
| SASRec + ALS simple blend | ... |
| Candidate union + rank blend | ... |
| Two-stage CatBoost/LightGBM | ... |

Дополнительно сохранить Candidate Recall@100/200.

---

## 10. Full-data training для Kaggle

После выбора ranker:

### Ranker
Ranker нельзя обучить на Kaggle hidden labels, поэтому использовать temporal training examples из known history.

Допустимый pragmatic approach:

- ranker обучается на 80→10 historical task;
- base candidate generators для final Kaggle inference refit'ятся на всех available events;
- features для Kaggle candidates строятся по full observed events;
- frozen ranker применяется к full-data candidate union.

При желании можно использовать несколько historical temporal cutoffs для ranker training, но только если реализация остаётся простой и leakage-safe.

---

## 11. Artifacts

Сохранить:

### Candidate recall diagnostics
`.artifacts/005_candidate_recall.json`

### Ranker training/validation metrics
`.artifacts/005_two_stage_validation.json`

### Optional candidate tables
В `.artifacts/candidates/`, желательно parquet.

### Ranker model
`.artifacts/models/5_two_stage_ranker.*`

### Submission
Следующий номер:

`.artifacts/submissions/5_submission_two_stage.csv`

Не перезаписывать старые submissions.

---

## 12. Kaggle

Не загружать автоматически.

В report вернуть путь к submission.

После ручной отправки Kaggle score должен быть записан в:
- `notes/hypotheses.md`
- `AGENTS.md`, если это новый confirmed best.

---

## 13. Что НЕ делать

В этом task не добавлять:

- BERT4Rec;
- GPT4Rec;
- новый transformer;
- большой hyperparameter search SASRec;
- custom ranking algorithm;
- сотни новых features.

Сначала завершить обязательную two-stage систему.

---

## Definition of Done

1. реализован leakage-safe 80/10/10 protocol для ranker;
2. candidate generators: SASRec + ALS + ItemKNN + popularity;
3. посчитан Candidate Recall для individual sources и unions;
4. создан unified candidate table;
5. features строятся без leakage;
6. есть simple blend/rank baseline;
7. обучен CatBoostRanker или LightGBM ranker;
8. two-stage Recall@10 сравнен с SASRec single-stage;
9. сохранён ranker checkpoint/model;
10. создан `5_submission_two_stage.csv`;
11. обновлены hypotheses/report;
12. final report явно отвечает:
   - насколько union поднял candidate recall;
   - добавляют ли ALS/KNN что-то к SASRec;
   - помог ли learned reranker;
   - local Recall@10 relative to SASRec 0.130385;
   - рекомендуемый следующий experiment.
