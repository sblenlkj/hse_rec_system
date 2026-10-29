# Task 04 — SASRec via RePlay, local benchmark + Colab-ready training

## Цель

Подготовить и провести первый полноценный transformer-based RecSys experiment, используя **SASRec** — одну из моделей, прямо рекомендованных преподавателем.

Главная цель — получить воспроизводимый pipeline:

```text
existing split
→ SASRec preprocessing
→ training
→ local Recall@10
→ top-N candidates + scores/ranks
→ full-data refit
→ checkpoint/weights
→ numbered Kaggle submission
```

Pipeline должен быть пригоден и для локального запуска на MacBook M1 Pro 16 GB, и для Google Colab GPU.

---

## 0. Обязательно прочитать

Перед изменениями:

- `AGENTS.md`;
- `notes/engineering.md`;
- `notes/data.md`;
- `notes/hypotheses.md`;
- Task 03 report;
- существующий split / metrics / submission code.

Не менять validation protocol.

---

## 1. Текущая reference point

Лучший classical result:

- rating-weighted ALS;
- local chronological Recall@10 = **0.067046**;
- Kaggle Recall@10 = **0.0819**;
- submission №3.

SASRec должен сравниваться именно с этим baseline.

---

## 2. Использовать готовую библиотеку

Не писать SASRec с нуля.

Первый выбор — **RePlay**, как рекомендовал преподаватель:

https://github.com/sb-ai-lab/RePlay/blob/main/examples/09_sasrec_example.ipynb

Если current RePlay API отличается от примера, использовать актуальную библиотечную реализацию и задокументировать это.

Если RePlay технически не работает на локальном Mac, не переписывать SASRec вручную:
- подготовить Colab notebook;
- использовать GPU;
- продолжить тем же experiment protocol.

---

## 3. Сначала local smoke test

На MacBook M1 Pro 16 GB сначала выполнить короткий smoke test, чтобы оценить feasibility.

Smoke test должен:
- использовать малое число epochs / batches;
- проверить полный preprocessing;
- запустить forward/train loop;
- получить хотя бы технически валидные recommendations;
- измерить примерное время epoch.

Цель smoke test — решить:

### Если estimated full training <= ~30–45 минут
Можно выполнить полноценный experiment локально.

### Если существенно дольше / MPS unsupported / dependency issues
Не тратить время на локальную борьбу. Перейти к Colab GPU.

Не уменьшать датасет для финальной метрики только ради локального запуска.

---

## 4. Colab notebook

Создать один notebook:

`notebooks/04_sasrec_colab.ipynb`

Он должен быть рассчитан на **Run All**.

Notebook должен:

1. проверить наличие GPU;
2. установить/pin необходимые зависимости;
3. подключить/получить проектный код;
4. загрузить подготовленные data files;
5. воспроизвести chronological 90/10 split;
6. подготовить sequential dataset;
7. обучить SASRec;
8. посчитать local Recall@10;
9. сохранить checkpoint;
10. получить top-N candidates + scores/ranks;
11. refit/final training на full events;
12. сгенерировать следующий numbered submission;
13. дать ссылки/пути на artifacts, которые нужно скачать обратно.

Не использовать скрытое состояние Colab, которое невозможно воспроизвести.

---

## 5. Данные для Colab

Не требовать от пользователя вручную преобразовывать CSV.

Подготовить минимальный понятный upload workflow.

Допустимые варианты:

### Вариант A — загрузить исходные competition CSV
Notebook сам читает:
- events.csv;
- item_features.csv;
- user_features.csv;
- submission_sample.csv.

### Вариант B — подготовить transformer-ready artifact локально
Если RePlay требует специфический format, добавить отдельный deterministic preprocessing script, который создаёт компактный artifact в `.artifacts/transformer_data/`.

Но preprocessing должен быть воспроизводим из raw data.

Нельзя коммитить competition data в git.

---

## 6. Validation

Основной protocol остаётся **chronological 90/10 per user**.

SASRec должен использовать только train history при local validation.

Запрещено:
- строить sequences по full history до split;
- включать validation items в training sequences;
- использовать validation для fitting encoders / mappings, если этого можно избежать;
- использовать full events для seen filtering при local evaluation.

Добавить sanity checks на sequence construction.

---

## 7. SASRec configuration

Не делать большой hyperparameter search.

Нужна одна разумная конфигурация на основе RePlay example/current docs.

Зафиксировать:
- max sequence length;
- hidden/embedding dimension;
- number of blocks/layers;
- heads;
- dropout;
- batch size;
- optimizer / lr;
- epochs;
- seed.

Если возможен early stopping — использовать validation, но не превращать это в sweep.

---

## 8. Что сохранить после обучения

Обязательно сохранить:

### Checkpoint / weights
В:
`.artifacts/models/`

Например:
`4_sasrec_checkpoint.*`

Checkpoint должен быть пригоден для повторной загрузки и inference.

### Validation result
Versioned artifact с:
- local Recall@10;
- runtime;
- device;
- hyperparameters;
- train/validation sizes;
- library/version;
- checkpoint path.

### Candidates
Сохранить минимум top-100, лучше top-200 candidates на пользователя с:
- user_id;
- item_id;
- sasrec_score;
- sasrec_rank.

Это понадобится для Task 05 two-stage recommender.

### Submission
Следующий свободный номер:

`.artifacts/submissions/4_submission_sasrec.csv`

Если по факту следующий номер уже занят, взять следующий свободный номер. Номер не переиспользовать.

---

## 9. Local vs Colab artifacts

Если обучение выполнено в Colab, notebook должен в конце собрать downloadable artifacts:

- checkpoint;
- validation JSON;
- top-N candidates;
- submission CSV.

Предпочтительно упаковать их в один zip для скачивания обратно в локальный проект.

После скачивания пользователь должен иметь возможность положить artifacts в соответствующие локальные директории без ручного преобразования.

---

## 10. Runtime decision

В report обязательно записать:

- local smoke-test runtime;
- device (CPU/MPS/CUDA);
- estimated or actual training time;
- Colab GPU type, если использовался;
- actual full training runtime.

Цель — понять, где дальше дешевле и быстрее экспериментировать.

---

## 11. Не делать в Task 04

Не делать:

- BERT4Rec одновременно с SASRec;
- GPT4Rec;
- CatBoost;
- two-stage ranking;
- широкий hyperparameter sweep;
- собственную SASRec implementation;
- сложный model serving.

Task 04 должен дать **один сильный и воспроизводимый transformer baseline**.

---

## 12. Подготовка к обязательному Task 05

Task 04 считается архитектурно успешным только если его output можно использовать как candidate source в two-stage recommender.

Поэтому top-N candidates, scores и ranks — обязательны, даже если Kaggle submission использует только top-10.

Task 05 будет использовать:

- SASRec candidates;
- ALS candidates;
- ItemKNN candidates;
- popularity fallback;
- engineered features;
- CatBoost/LightGBM reranking.

---

## Definition of Done

1. SASRec используется из готовой библиотеки;
2. выполнен local smoke test;
3. принято решение local vs Colab на основании runtime/compatibility;
4. есть Run All Colab notebook;
5. chronological validation не нарушена;
6. получен local Recall@10;
7. результат сравнен с ALS 0.067046;
8. сохранён checkpoint;
9. сохранены top-100/200 candidates + score/rank;
10. создан numbered SASRec submission;
11. сохранён versioned validation artifact;
12. обновлён `notes/hypotheses.md`;
13. completion report содержит:
   - library/version;
   - hyperparameters;
   - local runtime;
   - Colab runtime/device, если использован;
   - local Recall@10;
   - artifact paths;
   - рекомендуемый следующий шаг: Task 05 two-stage.
