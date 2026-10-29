# Журнал гипотез и экспериментов

Этот файл является рабочим журналом команды. Он должен показывать путь от первоначального анализа до лучшего решения.

## Правило

Каждый существенный эксперимент оформляется до или сразу после запуска.

Необходимо отличать:
- **гипотезу** — почему изменение должно помочь;
- **эксперимент** — что именно мы изменили;
- **метрику** — чем проверяем;
- **результат** — что получили;
- **вывод** — принимаем или отвергаем гипотезу;
- **следующий шаг** — что проверяем дальше.

## Шаблон

### E00 — Название эксперимента

**Дата:**  
**Автор:**  
**Git commit:**  

**Контекст**  
Что уже известно к моменту запуска.

**Гипотеза**  
Если сделать X, то метрика Y должна измениться в направлении Z, потому что ...

**Изменение**  
Что конкретно отличается от предыдущего решения.

**Контроль / baseline**  
С чем сравниваем.

**Метрики**
- Local Recall@10:
- Candidate Recall@N, если применимо:
- Kaggle Recall@10, если был сабмит:

**Результат**  
Фактические значения и наблюдения.

**Вывод**  
- hypothesis accepted / rejected / inconclusive;
- почему;
- что узнали о данных или модели.

**Следующие гипотезы**
1. ...
2. ...

---

## Краткая таблица экспериментов

| ID | Гипотеза / изменение | Local Recall@10 | Kaggle Recall@10 | Статус | Вывод |
|---|---|---:|---:|---|---|
| E00 | Submission sanity check | — | — | Done | 6 040 users, ровно 10 уникальных unseen items |
| E01 | Global popularity | 0.036744 | — | Done | Рабочий, но слабый неперсональный baseline |
| E02 | Popularity по rating >= 4 | 0.036252 | — | Rejected | Фильтр снизил Recall@10 на этом split |
| E03 | Cosine ItemKNN, 200 neighbors | **0.055817** | **0.0579** | Accepted | Local и Kaggle score близки; персонализация дала +51.9% относительно E01 |
| E04 | `implicit` ItemKNN + positive genre affinity, alpha=0.25 | **0.057316** | — | Accepted | +0.001499 / +2.69% к E03; создан submission №2 |
| E05 | ALS binary, 64 factors | 0.064361 | — | Accepted | +0.008545 к ItemKNN; заметно сильнее классического CF baseline |
| E06 | ALS с rating confidence, 64 factors | **0.067046** | **0.0819** | Accepted | Лучший classical результат; почти baseline_5=0.0821 на Kaggle |
| E07 | RePlay SASRec, ordered implicit sequence | **0.130385** | **0.1889** | Accepted | Сильный скачок относительно ALS; текущая основная модель |
| E08 | SASRec + ALS + ItemKNN + popularity, CatBoostRanker | **0.144796** | — | Accepted | +0.012081 к SASRec на том же holdout; union Recall вырос до 0.758038 |

## Task 01 — фактические результаты первого end-to-end решения

**Дата:** 2026-10-29

**Protocol:** chronological 90/10 per user, train 802 016 / validation 92 133 событий.

**Метрика:** macro Recall@10 по 6 040 пользователям с ground truth.

**Запуск:** `uv run recsys-baseline`.

| Experiment | Model | Local Recall@10 | Runtime, s | Notes |
|---|---|---:|---:|---|
| E01 | Global popularity | 0.036744 | 3.52 | Counts и seen filtering только по train |
| E02 | Popularity, rating >= 4 | 0.036252 | 3.51 | Positive-only counts, seen filtering по всему train |
| E03 | Cosine ItemKNN, top-200 neighbors | **0.055817** | 4.46 | Binary feedback, sparse cosine, train only |

### Выводы

- E01 подтверждает корректность простого candidate pipeline, но одного popularity недостаточно.
- E02 отвергает гипотезу, что отбрасывание оценок ниже 4 автоматически улучшит top-10: результат на 0.000493 ниже E01.
- E03 подтверждает пользу персонализации: ItemKNN улучшил E01 на 51.9% относительно baseline.
- Финальный ItemKNN переобучен на всех 894 149 событиях за 4.67 s. Submission содержит 6 040 строк; для каждого пользователя исключены все items, уже встречавшиеся в полном events.
- Полный measured runtime проверочного запуска — 16.95 s, включая чтение данных, split, sanity checks, три validation-эксперимента, full-data refit и запись submission.

### Leakage checks

- для каждого пользователя `max(train.timestamp) < min(validation.timestamp)`;
- `(user_id, item_id)` не пересекаются между train и validation;
- popularity, similarity, seen sets и рекомендации в local validation строятся только из train;
- validation используется только как ground truth;
- full events используются только после завершения local evaluation для final refit.

### Первый Kaggle submission

**Submission:** `submission_itemknn.csv`

**Kaggle Recall@10:** **0.0579**

Local chronological Recall@10 для той же модели: **0.055817**.

Разница небольшая (~0.0021 absolute), поэтому текущий chronological 90/10 protocol пока выглядит как разумный локальный proxy. Одного сабмита недостаточно, чтобы считать это доказанным, но сильного расхождения между local и public leaderboard не наблюдается.

### Следующий шаг

Проверить ALS на том же chronological split и параллельно сделать diagnostic masked-10% split. Сравнить ranking моделей на двух локальных protocol и, если ALS выглядит сильнее ItemKNN, сделать следующий Kaggle submission.

## Task 02 — train-only features и library-based CF

**Дата:** 2026-10-29

**Protocol:** chronological 90/10 per user, train 802 016 / validation 92 133 событий. Все derived features, genre profiles, similarity и candidate scores рассчитаны только по train. Source fingerprint проверен отдельным leakage assertion.

**Запуск:** `uv run recsys-features-cf`

**Библиотека:** `implicit==0.7.3`, `CosineRecommender(K=200)`, binary feedback, один thread.

### Feature layer

- User: interaction count, mean/std rating, count и fraction rating >= 4, распределение ratings 1–5.
- Item: interaction count, mean/std rating, count и fraction rating >= 4, 18 raw genre indicators.
- User × genre: L2-normalized rating-weighted и positive-only (rating >= 4) profiles.
- В content и hybrid ranking фактически использован positive-only profile. Rating-weighted profile сохранён в reusable layer, но в этом минимальном search не использовался для model selection.

### Результаты

| Experiment | Local Recall@10 | Runtime, s | Notes |
|---|---:|---:|---|
| Custom cosine ItemKNN rerun | 0.055817 | 4.43 | Task 01 reference, top-200 |
| `implicit` cosine ItemKNN | 0.055680 | 1.30 | top-200, top-100 candidates |
| Positive genre-profile content model | 0.014010 | 1.28 | cosine user-profile × item genres |
| `implicit` KNN + genre, alpha=0.10 | 0.056435 | 0.09 | min-max score blend over 100 KNN candidates |
| `implicit` KNN + genre, alpha=0.25 | **0.057316** | 0.09 | best hybrid |
| `implicit` KNN + genre, alpha=0.50 | 0.056007 | 0.09 | genre signal starts to dominate |

Feature build занял 0.55 s; полный measured run вместе с full-data refit и записью submission — 10.62 s.

### Вывод

- Library ItemKNN практически воспроизводит custom baseline, но сам по себе ниже на 0.000136 Recall@10.
- Чистый genre content model слишком слаб как самостоятельный recommender.
- Genre affinity полезен как небольшой reranking signal: alpha=0.25 даёт +0.001499 absolute / +2.69% относительно Task 01 reference.
- Более сильный genre weight ухудшает результат, поэтому расширять search сейчас не нужно.
- Лучший hybrid переобучен на full events; все observed items исключены. Создан `.artifacts/submissions/2_submission_library_genre_hybrid.csv`. Kaggle score пока неизвестен.

### Следующий шаг

Проверить ALS на том же split и сравнить его не только по Recall@10, но и как complementary candidate source для ItemKNN + genre hybrid.

## Task 03 — ALS и diagnostic masked validation

**Дата:** 2026-10-29

**Основной protocol:** chronological 90/10 per user, train 802 016 / validation 92 133 interactions. Это тот же split и те же leakage invariants, что в Task 01–02.

**Diagnostic protocol:** deterministic masked-10% per user с seed=42, train 802 016 / validation 92 133 interactions. User-item pairs не пересекаются; fitting и seen filtering используют только masked train. Этот protocol не заменяет chronological.

**Запуск:** `PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m recsys.als_pipeline`

**Библиотека:** `implicit==0.7.3`, CPU ALS, 64 factors, regularization=0.01, 15 iterations, один thread, random_state=42.

### Гипотезы

1. Low-rank ALS лучше ItemKNN моделирует устойчивые скрытые предпочтения и повысит chronological Recall@10.
2. Rating 1–5 как confidence даст более полезный ranking, чем одинаковый binary weight.
3. Если masked split ближе к предполагаемому hidden-set construction, абсолютные score могут измениться; без дополнительных Kaggle observations это остаётся только diagnostic.

### Результаты

| Model | Chronological Recall@10 | Masked Recall@10 | Chronological runtime, s | Masked runtime, s |
|---|---:|---:|---:|---:|
| Popularity | 0.036744 | 0.069072 | 3.49 | 3.48 |
| ItemKNN cosine, 200 neighbors | 0.055817 | 0.109801 | 4.40 | 4.45 |
| ALS binary | 0.064361 | 0.156817 | 2.02 | 2.03 |
| ALS rating-weighted | **0.067046** | **0.169748** | 2.77 | 2.81 |

Полный measured run вместе с full-data refit и записью submission занял 30.26 s.

### Вывод

- Обе ALS-конфигурации заметно сильнее ItemKNN. Rating-weighted ALS улучшил Task 01 ItemKNN на 0.011229 absolute и лучший Task 02 hybrid на 0.009730 absolute.
- Raw rating 1–5 как confidence улучшил binary ALS на 0.002685 chronological Recall@10, поэтому E06 принят как новый local best.
- Ranking моделей совпал на обоих split: ALS weighted > ALS binary > ItemKNN > Popularity. Это укрепляет вывод о сравнительном преимуществе ALS.
- Masked scores существенно выше chronological scores для всех моделей. Сравнивать их абсолютные значения напрямую нельзя: это разные задачи. Одного совпадения ranking недостаточно, чтобы утверждать, что masked split ближе к Kaggle hidden set; chronological остаётся основным protocol, поскольку для ItemKNN уже есть близкий local/public score.
- Optional blend отложен: ALS уже дал крупное улучшение, а дополнительный эксперимент не нужен для ответа на вопросы Task 03.

### Submission

Лучший rating-weighted ALS переобучен на всех 894 149 interactions. Для каждого пользователя исключены все items из full history.

**Файл:** `.artifacts/submissions/3_submission_als.csv`

**Kaggle Recall@10:** **0.0819**.

Это почти baseline_5 = 0.0821 и подтверждает, что ALS существенно сильнее ItemKNN не только локально, но и на leaderboard.

### Следующий шаг

Перейти к SASRec на неизменном chronological split; использовать ALS как основной classical reference.

## Task 04 — RePlay SASRec

**Дата:** 2026-10-29

**Protocol:** chronological 90/10 per user, train 802 016 / validation 92 133 interactions. Item vocabulary, sequences, fitting, candidate generation и local seen filtering построены только по train. Validation используется как ground truth; неизвестные train vocabulary validation items не попадают в model fitting.

**Запуск:** `PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m recsys.sasrec_pipeline --mode full --device cpu`

**Библиотеки:** `replay-rec==0.22.0`, `torch==2.15.0`, `lightning==2.6.6`. Использован актуальный block-based RePlay API (`ParquetModule`, `SasRec.from_params`, Lightning wrapper), а не собственная реализация SASRec.

### Конфигурация

- max sequence length 50, embedding dimension 64;
- 2 transformer blocks, 2 attention heads, dropout 0.3;
- batch size 128, Adam, learning rate 0.001;
- максимум 10 epochs, early stopping patience 2, seed 42;
- model input — упорядоченные item ids; rating не подаётся в SASRec.

### Runtime decision

Smoke test использовал полный preprocessing, 8 train batches и 4 validation batches. На CPU он занял 2.24 s и оценил 10 epochs примерно в 134.20 s, что существенно ниже порога 30–45 минут. MPS был скомпилирован, но недоступен в текущем runtime, поэтому полноценный эксперимент выполнен локально на CPU. Colab GPU не использовался; для будущего воспроизводимого GPU-запуска подготовлен `notebooks/04_sasrec_colab.ipynb`.

### Результат

| Model | Local Recall@10 | Difference vs ALS | Validation training, s |
|---|---:|---:|---:|
| Rating-weighted ALS | 0.067046 | — | 2.77 |
| RePlay SASRec | **0.130385** | **+0.063339** | 153.87 |

Лучшей оказалась 10-я эпоха (zero-based epoch 9), поэтому full-data refit также выполнен на 10 epochs. Он занял 102.41 s; полный measured runtime validation + inference + refit + artifacts составил 277.27 s.

### Артефакты и вывод

- validation checkpoint: `.artifacts/models/4_sasrec_validation_best.ckpt`;
- full-data checkpoint: `.artifacts/models/4_sasrec_checkpoint.ckpt`;
- validation и full-data top-200: `.artifacts/candidates/4_sasrec_validation_top200.csv` и `.artifacts/candidates/4_sasrec_full_top200.csv`;
- versioned metrics: `.artifacts/004_sasrec_validation.json`;
- submission №4: `.artifacts/submissions/4_submission_sasrec.csv`;
- переносимый bundle: `.artifacts/4_sasrec_artifacts.zip`.

E07 принят: последовательная модель более чем вдвое улучшила local Recall@10 относительно ALS. Kaggle Recall@10 = **0.1889**, что подтверждает сильный перенос uplift на leaderboard. Следующий обязательный эксперимент — Task 05 candidate union / two-stage reranking с SASRec, ALS, ItemKNN и popularity.

## Task 05 — leakage-safe candidate union и CatBoost reranking

**Дата:** 2026-10-29

**Protocol:** прямой chronological 80/10/10 per user. Candidate generators и ranking features для обучения ranker fit только на первых 80%; следующие 10% используются только как labels. Для final evaluation generators и features заново строятся по первым 90%, frozen ranker применяется к последним 10%. Размеры: 712 916 / 89 100 / 92 133 interactions.

**Библиотеки:** RePlay SASRec 0.22.0, `implicit` 0.7.3 ALS/ItemKNN, CatBoostRanker 1.2.10 с YetiRank. SASRec обучался фиксированные 10 epochs без early stopping по будущему target; ranker использует 300 деревьев depth 7, learning rate 0.05, L2=5, seed 42.

### Candidate recall на final 10%

| Candidate source | Candidate Recall |
|---|---:|
| SASRec@50 | 0.348203 |
| SASRec@100 | 0.471320 |
| SASRec@200 | 0.598884 |
| ALS@100 / @200 | 0.395648 / 0.553696 |
| ItemKNN@100 / @200 | 0.292925 / 0.449053 |
| SASRec ∪ ALS | 0.718612 |
| SASRec ∪ ALS ∪ ItemKNN | 0.756941 |
| Все источники + popularity | **0.758038** |

Classical models дают содержательно complementary candidates: ALS добавляет 12 845 relevant user-item pairs вне SASRec top-200, ItemKNN — 11 815, popularity — 4 043. ALS повышает candidate recall на 0.119728 absolute, ItemKNN поверх SASRec+ALS ещё на 0.038329; popularity добавляет только 0.001097.

### Final Recall@10

| Ranking | Recall@10 |
|---|---:|
| SASRec single-stage, fixed 10 epochs | 0.132715 |
| SASRec + ALS normalized score blend | 0.126825 |
| SASRec + ALS + KNN reciprocal-rank blend | 0.104150 |
| Candidate union + CatBoost YetiRank | **0.144796** |

Простые blends ухудшили ranking: scores источников даже после per-user normalization недостаточно сопоставимы, а равный reciprocal-rank blend переоценивает слабые sources. Learned ranker использовал 45 компактных train-only features и улучшил тот же final holdout на 0.012081 absolute относительно SASRec; относительно Task 04 reference 0.130385 прирост составляет 0.014411.

### Артефакты и вывод

- candidate diagnostics: `.artifacts/005_candidate_recall.json`;
- validation metrics: `.artifacts/005_two_stage_validation.json`;
- ranker: `.artifacts/models/5_two_stage_ranker.cbm`;
- candidate tables: `.artifacts/candidates/5_*_candidates.parquet`;
- submission: `.artifacts/submissions/5_submission_two_stage.csv`.

E08 принят: classical candidates действительно дополняют SASRec, и learned reranking превращает дополнительный recall в top-10 uplift. Submission №5 использует frozen historical ranker, full-data generators/features и исключает все observed items. Kaggle upload остаётся ручным; следующий эксперимент следует выбирать после leaderboard score №5, приоритетно проверить calibration/distribution shift ranker либо один дополнительный historical cutoff, а не расширять feature set вслепую.

## Примеры хороших гипотез

Хорошо:

> Добавление SASRec candidates к ALS повысит Candidate Recall@200, потому что SASRec использует порядок недавних взаимодействий, а ALS моделирует более стабильное долгосрочное предпочтение.

Плохо:

> Попробуем SASRec.

Хорошо:

> Использование rating >= 4 как positive feedback повысит Recall@10 относительно бинаризации всех interactions, если низкие оценки действительно выражают отрицательное предпочтение.

Плохо:

> Попробуем удалить низкие оценки.

Хорошо:

> CatBoost reranking candidate union повысит Recall@10 относительно фиксированного score blending, если generator scores, user/item features и source flags содержат complementary signal.

## Важное правило честности экспериментов

Все локальные результаты должны быть получены на одном согласованном temporal validation protocol.

Если split или preprocessing меняется, эксперимент нельзя напрямую сравнивать с предыдущими без специальной пометки.
