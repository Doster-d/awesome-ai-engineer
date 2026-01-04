# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: 61x2500 (строк x столбцов)
- Целевая переменная: `target` (False - 95%, True - 5%)
- Признаки: f1-f60 числа с плавающей точки.

## 2. Protocol

- Разбиение: train/test (0.8/0.2 (train (20000, 60), test (5000, 60)), `random_state`=42, стратификация)
- Подбор: **GridSearchCV на train**, CV = **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)**.
  Оптимизировали **ROC-AUC** (`scoring="roc_auc"`), `refit=True`. Test использовали один раз для финальной оценки.
- Метрики:
  - accuracy - средния оценка, базовая поддерживая всеми моделями
  - F1 - важен баланс precision/recall при дисбалансе классов
  - ROC-AUC - основная метрика для выбора лучшей модели, устойчивая к дисбалансу классов

## 3. Models

Опишите, какие модели сравнивали и какие гиперпараметры подбирали.

Минимум:

- **DummyClassifier** (baseline)
  - `strategy="most_frequent"` (всегда предсказывает мажоритарный класс).
- **LogisticRegression** (baseline из S05)
  - Pipeline: `RobustScaler` + `LogisticRegression(max_iter=4000)`.
  - Подбор (GridSearchCV, scoring=roc_auc):
    - `C`: [0.1, 1.0, 10.0]
    - `penalty`: ["l2"]
    - `solver`: ["lbfgs"]
- **DecisionTreeClassifier** (контроль сложности)
  - Pipeline: `RobustScaler` + `DecisionTreeClassifier`.
  - Подбор:
    - `max_depth`: [None, 3, 5, 8]
    - `min_samples_leaf`: [1, 5, 10, 20]
    - `ccp_alpha`: [0.0, 0.001, 0.005, 0.01]
**RandomForestClassifier**
- Pipeline: `RobustScaler` + `RandomForestClassifier(n_estimators=100, n_jobs=-1)`.
- Подбор:
  - `max_depth`: [None, 6, 10]
  - `min_samples_leaf`: [1, 5, 10]
  - `max_features`: ["sqrt", 0.5]

- **HistGradientBoostingClassifier** (boosting)
  - Pipeline: `RobustScaler` + `HistGradientBoostingClassifier(early_stopping=True)`.
  - Подбор:
    - `learning_rate`: [0.03, 0.05, 0.1]
    - `max_depth`: [2, 3, None]
    - `max_leaf_nodes`: [15, 31, 63]

- **StackingClassifier** (ансамбль поверх лучших моделей)
  - Базовые модели: лучший `LogReg(scaled)`, лучший `RandomForest`, лучший `HistGradientBoosting`.
  - `final_estimator`: `LogisticRegression(max_iter=4000)`
  - Внутренний CV стекинга: `cv=5`, `n_jobs=-1`, `passthrough=False`.

## 4. Results


Финальные метрики на test (как сохранено в `artifacts/metrics_test.json`):

- Stacking: accuracy=0.9798, F1=0.7543, ROC-AUC=0.8140
- HistGradientBoosting: accuracy=0.9772, F1=0.7031, ROC-AUC=0.7741
- DecisionTree: accuracy=0.9672, F1=0.5900, ROC-AUC=0.7361
- RandomForest: accuracy=0.9684, F1=0.5298, ROC-AUC=0.6808
- LogReg(scaled): accuracy=0.9632, F1=0.4286, ROC-AUC=0.6395
- Dummy(most_frequent): accuracy=0.9508, F1=0.0000, ROC-AUC=0.5000

Победитель: **Stacking** (лучший по ROC-AUC и F1 на test среди сравниваемых).  
Интерпретация: стекинг агрегирует разные “взгляды” на данные (линейная модель + bagging + boosting), что даёт более устойчивое качество на дисбалансной задаче.

## 5. Analysis

- Устойчивость (random_state):
  - Отдельный эксперимент с 5 разными `random_state` в текущей версии не проводился (в задании это опционально). Везде использовался фиксированный `random_state=42`.

- Ошибки (confusion matrix для лучшей модели — Stacking, test):
  Матрица ошибок (true по строкам, pred по столбцам):
  - TN=4744, FP=10
  - FN=91,  TP=155

  Комментарий:
  - Очень мало FP (10) → высокая precision по положительному классу.
  - FN заметно больше (91) → recall по положительному классу умеренный.

- Интерпретация (permutation importance, top-15 для Stacking по падению ROC-AUC):
  Топ признаков: **f54, f25, f58, f53, f04, f33, f38, f47, f41, f13, f08, f29, f36, f43, f57**.

- Вывод:
  - Наиболее важны несколько признаков (лидер — f54), но вклад распределён по группе факторов.
  - Перестановочная важность показывает чувствительность качества к “разрушению” конкретного признака; это практичная, модель-агностичная интерпретация для ансамбля.

## 6. Conclusion

- На дисбалансных данных accuracy легко “обманывает”: Dummy даёт ~0.95 accuracy, но F1=0.
- Контроль сложности дерева (min_samples_leaf / max_depth / ccp_alpha) критичен, иначе дерево переобучается.
- RandomForest снижает variance по сравнению с одиночным деревом, но качество по редкому классу всё равно зависит от порога и информативности признаков.
- Boosting и особенно stacking дают выигрыш, комбинируя разные типы моделей и уменьшая ошибки отдельных алгоритмов.
- Честный протокол: подбирать гиперпараметры только на train через CV, test использовать один раз для финальной оценки, фиксировать random_state.
- Для ROC-AUC в бинарной задаче корректно считать метрику по вероятностям (predict_proba), а не по жёстким предсказаниям.
