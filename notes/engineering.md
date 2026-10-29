# Engineering principles

Этот документ задаёт постоянные инженерные правила проекта. От них не следует отходить без явной причины.

## 1. Leakage safety — главный инвариант

Любая модель, feature engineering, preprocessing или candidate generation должны быть построены так, чтобы validation/test не влияли на обучение.

Основной local protocol:

- история каждого пользователя сортируется по `timestamp`;
- первые 90% interactions -> train;
- последние 10% -> validation.

Только train может использоваться для построения:

- popularity;
- user aggregates;
- item aggregates;
- user-item statistics;
- genre preferences;
- similarities;
- embeddings;
- matrix factorization;
- candidate generators;
- ranking features;
- normalization/statistics;
- learned encoders.

Validation используется только как ground truth.

### Типичные ошибки, которых нельзя допускать

- считать item popularity по полному `events`;
- считать средний rating фильма с использованием validation;
- строить user genre profile по будущим событиям;
- fit scaler/encoder по train+validation;
- строить similarity matrix по полному датасету;
- обучать embeddings до split;
- использовать full events для seen filtering в local validation;
- подбирать параметры по public leaderboard вместо local validation.

Перед каждым новым экспериментом нужно проверить, где именно считается каждая статистика и на какой части данных она fit'ится.

---

## 2. Готовые библиотеки предпочтительнее самописных реализаций

Если существует зрелая библиотека, которая корректно решает задачу, использовать её предпочтительнее, чем писать алгоритм с нуля.

Причины:

- меньше вероятность математических и численных ошибок;
- проще воспроизводимость;
- быстрее экспериментальный цикл;
- легче объяснить решение;
- меньше собственного кода для поддержки.

Самописная реализация оправдана только если:

- библиотека не подходит по API/окружению;
- нужно проверить конкретную исследовательскую гипотезу;
- реализация настолько проста, что библиотека была бы тяжелее самого решения;
- нужен специальный behavior, которого нет в библиотеке.

### Предпочтительные библиотеки

Для классического RecSys по возможности рассматривать:

- **implicit** — ALS / nearest-neighbour collaborative filtering;
- **RePlay** — рекомендованные преподавателем RecSys pipelines, SASRec/BERT4Rec и другие модели;
- **scikit-learn** — preprocessing, nearest neighbours, metrics/helpers;
- **LightGBM / CatBoost** — если дойдём до ranking;
- **PyTorch** — neural / transformer experiments.

Не реализовывать вручную ALS, ItemKNN, BERT4Rec, SASRec или стандартные similarity methods без необходимости.

Если существующая самописная версия уже есть, её можно оставить как baseline/reference, но новые эксперименты должны по возможности использовать библиотечную реализацию.

---

## 3. Feature engineering должен быть train-only и проверяемым

Raw features в датасете ограничены:

### User
- gender;
- age bucket.

### Item
- 18 genre indicators.

Основной потенциал находится в features, построенных из interaction history.

### User features

- interaction count;
- positive interaction count;
- mean/std rating;
- recency/activity;
- distribution over ratings;
- genre affinity;
- fraction of interactions per genre;
- demographic + behavioral features.

### Item features

- interaction count / popularity;
- positive interaction count;
- average/std rating;
- rating distribution;
- recent popularity;
- genre indicators;
- demographic audience statistics, если это leakage-safe.

### User-item features

- interaction / repeat indicators, если постановка позволяет;
- genre affinity между user profile и item genres;
- similarity;
- model score;
- model rank;
- source flags;
- latent-factor similarities.

### Правило

Каждая derived feature должна иметь ясный ответ на вопрос:

> Из каких именно train events она построена?

Если этот ответ неочевиден, feature считается подозрительной на leakage.

---

## 4. Feature engineering не должен подменять модель

Не нужно сразу строить 200–300 признаков.

Сначала:

1. создать небольшой осмысленный набор;
2. проверить, даёт ли он измеримый прирост;
3. оставить полезные признаки;
4. только затем расширять.

Мы не оптимизируем количество признаков. Мы проверяем гипотезы.

---

## 5. Transformer — обязательное направление проекта

Преподаватель отдельно рекомендовал попробовать сильные sequential transformer models:

- SASRec;
- BERT4Rec;
- при необходимости GPT4Rec.

Для SASRec/BERT4Rec преподаватель рекомендовал **RePlay**:

https://github.com/sb-ai-lab/RePlay/blob/main/examples/09_sasrec_example.ipynb

Поэтому проект должен прийти к обучению как минимум одного сильного transformer-based recommender.

Это не optional "если останется время", а одна из основных исследовательских веток.

При этом:

- transformer должен использовать тот же leakage-safe validation protocol;
- его score должен сравниваться с classical baselines;
- желательно использовать готовую реализацию, а не писать transformer с нуля;
- если локальное Mac-окружение мешает, использовать Google Colab GPU;
- обучение transformer не должно ломать общую архитектуру проекта.

Целевой путь:

```text
clean validation
→ classical baselines
→ feature engineering
→ library-based CF models
→ SASRec/BERT4Rec
→ сравнение
→ простой ensemble / two-stage approach при наличии времени
```

---

## 6. Воспроизводимость

Каждый существенный experiment должен фиксировать:

- model/library/version;
- split;
- preprocessing;
- hyperparameters;
- random seed;
- local Recall@10;
- runtime;
- Kaggle score, если был submission;
- git commit / experiment id.

Итоговое решение должно запускаться одной командой или одним полным notebook run.

---

## 7. Минимум собственного кода

Собственный код проекта должен в основном отвечать за:

- data loading;
- split;
- leakage checks;
- feature building;
- orchestration;
- common evaluation;
- submission formatting;
- experiment logging.

Алгоритмические модели по возможности импортируются из библиотек.

---

## 8. Принцип выбора следующего эксперимента

Следующий experiment должен отвечать хотя бы на один вопрос:

- улучшает ли качество?
- добавляет ли complementary candidates?
- улучшает ли correlation local ↔ Kaggle?
- помогает ли specific user/item segment?
- подтверждает ли гипотезу о sequential signal?
- полезны ли engineered features?

Если эксперимент не отвечает ни на один из этих вопросов, его не нужно делать.
