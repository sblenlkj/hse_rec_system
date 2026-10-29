# План UV RecSys

## 0. Цель

Построить воспроизводимое решение HSE RecSys Challenge 2026 с качеством выше требуемого baseline по **Recall@10**, при этом сохранить понятный путь экспериментов и уметь объяснить каждое существенное решение.

Мы заранее ведём две ветки:

1. **быстрый сильный baseline** — в том числе SASRec / BERT4Rec;
2. **non-transformer / hybrid pipeline** — collaborative filtering, candidate generation, features, boosting, ensembles.

Так мы не зависим от возможного дополнительного требования преподавателя отказаться от трансформера.

---

## 1. Данные и EDA

Файлы:
- `events.csv`;
- `user_features.csv`;
- `item_features.csv`;
- `submission_sample.csv`.

Проверить:

### Events
- число interactions;
- число пользователей;
- число фильмов;
- колонки и типы;
- временной диапазон;
- распределение числа interactions на пользователя;
- распределение числа interactions на item;
- sparsity user-item матрицы;
- распределение ratings;
- дубликаты user-item;
- повторные оценки;
- долю пользователей с короткой историей;
- cold / warm users и items.

### User features
- набор признаков;
- пропуски;
- cardinality категориальных признаков;
- покрытие пользователей относительно events.

### Item features
- genres / categories / textual или другие признаки;
- пропуски;
- покрытие items относительно events;
- возможность построить content similarity и user preference features.

---

## 2. Валидация

### Основной protocol

`timestamp` не является общей календарной шкалой: это относительный порядок внутри пользовательской истории.

Поэтому основной split:

1. сортируем события каждого пользователя по `timestamp`;
2. первые 90% истории -> train;
3. последние 10% -> validation.

Этот protocol фиксируем для сравнения всех первых моделей.

### Leakage invariants

Только train-side данные используются для:
- user/item statistics;
- popularity;
- similarity;
- ALS factors;
- sequential histories;
- candidate generation;
- ranking features.

Validation используется только как ground truth.

Любая функция обучения/feature engineering должна получать train явно, а не полный `events`.

### Дополнительный diagnostic split

В данных около 10% пропущенных timestamp-позиций внутри историй. Это может означать masked hidden interactions в Kaggle test, но пока это лишь гипотеза.

После получения нескольких честных baseline можно отдельно проверить masked-10% validation как diagnostic protocol и сравнить направление изменений с Kaggle. Основной chronological split при этом сохраняется.

### Локальная метрика

Реализуем `Recall@10` максимально близко к Kaggle и отдельно фиксируем правила:
- macro/micro;
- пользователи без ground truth;
- seen-item filtering;
- отсутствие duplicate recommendations.

---

## 3. Baseline ladder

Цель этапа — не максимизировать качество, а построить проверяемую лестницу моделей.

### E00 — Submission sanity check
- валидный формат сабмита;
- 10 items на пользователя;
- отсутствие дубликатов;
- coverage test users.

### E01 — Global popularity
Топ-10 наиболее популярных фильмов по train.

### E02 — Rating-aware popularity
Проверить несколько простых вариантов:
- interaction count;
- средняя оценка;
- weighted popularity;
- positive-only popularity.

### E03 — Item similarity / ItemKNN
Проверить:
- cosine;
- co-occurrence;
- rating-aware similarity.

### E04 — ALS
Несколько разумных вариантов:
- implicit interactions;
- rating-weighted interactions;
- фильтрация / weighting low ratings.

### E05 — SASRec
RePlay baseline.

Цель: проверить утверждение преподавателя о сильном transformer fit-predict baseline.

### E06 — BERT4Rec
Только если SASRec показывает, что sequential branch действительно перспективна.

GPT4Rec пока не является обязательным этапом: сначала проверяем более дешёвые sequential models.

---

## 4. Candidate generation

После отдельных baseline переходим от соревнования моделей к их сочетанию.

Candidate sources:
- Popularity;
- ItemKNN;
- ALS;
- SASRec;
- BERT4Rec;
- content-based candidates при наличии полезных item features.

Для каждого генератора сохраняем:
- `score`;
- `rank`;
- `source`;
- top-N candidates.

### Главная диагностическая метрика

До reranking считаем **Candidate Recall@N**.

Пример:

```text
ALS                 Recall@200 = ...
KNN                 Recall@200 = ...
SASRec              Recall@200 = ...
ALS ∪ KNN           Recall@200 = ...
ALS ∪ KNN ∪ SASRec  Recall@200 = ...
```

Это позволит понять, какие слабые модели всё равно дают complementary candidates.

---

## 5. Feature engineering

После формирования candidate set строим признаки только на train-side относительно cutoff.

### User features
- число interactions;
- число positive interactions;
- средний rating;
- variance rating;
- activity / recency;
- любимые genres/categories;
- доля взаимодействий по жанрам;
- признаки из `user_features.csv`.

### Item features
- popularity;
- average rating;
- interaction count;
- recent popularity;
- genres/categories;
- признаки из `item_features.csv`.

### User-item features
- был ли item ранее у пользователя;
- число взаимодействий;
- last interaction time;
- user-category affinity;
- user-item / user-genre similarity;
- embedding similarity.

### Candidate-source features
- ALS score / rank;
- KNN score / rank;
- SASRec score / rank;
- BERT4Rec score / rank;
- был ли item найден каждым генератором;
- число генераторов, нашедших candidate.

---

## 6. Ranking

Проверить по возрастанию сложности:

1. один лучший candidate score;
2. hand-crafted weighted score;
3. score blending;
4. rank blending;
5. CatBoost / LightGBM ranker.

Не переходить к 200–300 признакам автоматически. Сначала получить небольшой осмысленный feature set и проверить, что reranking вообще даёт прирост.

---

## 7. Ensembles

Основные гипотезы:
- модели находят разные релевантные items;
- union повышает candidate recall;
- score/rank blending повышает Recall@10;
- CatBoost учится выбирать между источниками лучше фиксированных весов;
- transformer + classical CF дополняют друг друга.

Финальный ensemble строить только после стабильной локальной валидации.

---

## 8. Transformer vs non-transformer

Поддерживаем две независимые линии:

### Transformer
- SASRec;
- BERT4Rec;
- при необходимости GPT4Rec.

### Non-transformer
- popularity;
- similarity / KNN;
- ALS / MF;
- feature engineering;
- boosting;
- ensembles.

Это позволит удовлетворить возможному дополнительному условию преподавателя.

---

## 9. Работа с Kaggle

Не использовать public leaderboard как единственную валидацию.

Каждый существенный сабмит должен иметь:
- experiment id;
- git commit;
- локальную метрику;
- краткое описание изменения;
- Kaggle metric;
- вывод.

При лимите 10 submissions/day сабмитим только содержательные варианты.

---

## 10. Воспроизводимость

К финалу:

- фиксируем random seeds;
- фиксируем зависимости;
- исключаем ручные промежуточные шаги;
- итоговый pipeline запускается одной командой или одним notebook run;
- лучший сабмит можно получить заново из чистого состояния.

Переиспользуемый код помещаем в `src/uv_recsys/`.

Экспериментальные ноутбуки — в `notebooks/`.

---

## 11. Ближайшие действия

1. Разобрать schema всех CSV.
2. Провести EDA.
3. Зафиксировать chronological 90/10 per-user split.
4. Реализовать leakage-safe split.
5. Реализовать Recall@10.
6. Получить Popularity baseline.
7. Получить ALS.
8. Получить ItemKNN.
9. Получить SASRec через RePlay.
10. Сравнить local и Kaggle.
11. После первых результатов сформировать следующую волну гипотез.


---

## 12. Обязательный финальный маршрут

После Task 03 приоритет фиксирован:

### Task 04 — Transformer baseline
- SASRec через RePlay как первый выбор;
- тот же leakage-safe validation protocol;
- local Recall@10;
- submission;
- сравнение с classical best.

Если SASRec технически недоступен, использовать BERT4Rec или другую корректную библиотечную реализацию.

### Task 05 — Two-stage recommender
Обязательная архитектура:

```text
multiple candidate generators
→ union top-N candidates
→ model/user/item/user-item features
→ CatBoost/LightGBM reranker
→ top-10
```

Минимальный candidate pool:
- transformer;
- ALS;
- ItemKNN;
- popularity как fallback/source.

Минимальные ranking features:
- source flag;
- score и rank каждой модели;
- item popularity;
- genre affinity;
- несколько user/item aggregates.

Главная диагностическая метрика до ranker:
- Candidate Recall@N.

Главная итоговая метрика:
- Recall@10.

### Task 06 — optional final ensemble/tuning
Только после того, как transformer и two-stage pipeline уже реализованы.

Можно:
- BERT4Rec vs SASRec;
- простой blend;
- небольшой hyperparameter tuning;
- postprocessing.

Нельзя откладывать two-stage pipeline ради большого числа дополнительных single-model экспериментов.
