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
| E05 | SASRec | — | — | TODO | |
| E06 | Candidate union | — | — | TODO | |
| E07 | Feature reranking | — | — | TODO | |

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
