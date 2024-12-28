# Logistic_regression_project
Logistic_regression_project -  Классификация пола на основе роста и веса с использованием логистической регрессии

# Проект: Классификация пола на основе роста и веса с использованием логистической регрессии

## Описание проекта
Проект посвящен анализу данных демографии Howel1 и построению модели логистической регрессии для классификации пола на основе роста и веса. Данные были предварительно обработаны, разделены на обучающую и тестовую выборки, а также нормированы. Модель была обучена и протестирована, а ее качество оценено с помощью различных метрик.

## Использованные библиотеки
- **`numpy`**: Для работы с массивами и выполнения математических операций.
- **`pandas`**: Для загрузки, фильтрации и обработки данных в формате DataFrame.
- **`matplotlib.pyplot`**: Для визуализации данных (построение графиков).
- **`sklearn.model_selection.train_test_split`**: Для разделения данных на обучающую и тестовую выборки.
- **`sklearn.preprocessing.StandardScaler`**: Для нормировки данных (приведение к единому масштабу).
- **`sklearn.linear_model.LogisticRegression`**: Для обучения модели логистической регрессии.
- **`sklearn.metrics`**: Для оценки качества модели:
  - `accuracy_score` (точность классификации).
  - `confusion_matrix` (матрица ошибок).
  - `precision_score` (точность).
  - `recall_score` (полнота).
  - `fbeta_score` (F-мера).
  - `roc_curve` (ROC-кривая).
  - `roc_auc_score` (площадь под ROC-кривой).
  - `RocCurveDisplay` (визуализация ROC-кривой).
    
## Методы:
- **Загрузка данных** с помощью `pandas.read_csv`.
- **Фильтрация данных** по условию (`np.logical_and`).
- **Визуализация данных** с помощью `matplotlib.pyplot.scatter`.
- **Разделение данных** на обучающую и тестовую выборки (`train_test_split`).
- **Нормировка данных** (`StandardScaler`).
- **Обучение модели логистической регрессии** (`LogisticRegression`).
- **Предсказание на тестовых данных** (`predict`).
- **Оценка точности модели** (`accuracy_score`).
- **Построение матрицы ошибок** (`confusion_matrix`).
- **Расчет метрик качества**:
  - Точность (`precision_score`).
  - Полнота (`recall_score`).
  - F-мера (`fbeta_score`).
- **Построение ROC-кривой** и расчет площади под ней (`roc_curve`, `roc_auc_score`, `RocCurveDisplay`).

## Основные этапы проекта
1. **Загрузка данных**:
   - Данные загружены с помощью `pandas.read_csv` из удаленного CSV-файла.

2. **Предобработка данных**:
   - Фильтрация данных по возрасту (от 18 до 50 лет) с использованием `np.logical_and`.
   - Разделение данных на признаки (`height`, `weight`) и целевую переменную (`male`).

3. **Визуализация данных**:
   - Построение scatter-графика с использованием `matplotlib.pyplot.scatter` для визуализации зависимости роста и веса для мужчин и женщин.

4. **Разделение данных**:
   - Данные разделены на обучающую и тестовую выборки с помощью `train_test_split`.

5. **Нормировка данных**:
   - Признаки нормированы с использованием `StandardScaler` для приведения к единому масштабу.

6. **Обучение модели**:
   - Модель логистической регрессии обучена с использованием `LogisticRegression` с параметром `solver='lbfgs'`.

7. **Предсказание и оценка модели**:
   - Предсказание на тестовых данных с помощью `predict`.
   - Оценка точности модели с использованием `accuracy_score`.
   - Построение матрицы ошибок с помощью `confusion_matrix`.
   - Расчет метрик качества: точность (`precision_score`), полнота (`recall_score`), F-мера (`fbeta_score`).

8. **Визуализация результатов**:
   - Построение ROC-кривой с использованием `roc_curve` и `RocCurveDisplay`.
   - Расчет площади под ROC-кривой с помощью `roc_auc_score`.

## Примеры кода
```python
# Загрузка данных
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';')

# Фильтрация данных
df_sample = df[np.logical_and(df['age'] >= 18, df['age'] <= 50)]

# Визуализация данных
import matplotlib.pyplot as plt
def plot(df):
    plt.figure(figsize=(20,10))
    plt.scatter(df.weight[df.male == 1], df.height[df.male == 1], color='blue', label='male')
    plt.scatter(df.weight[df.male == 0], df.height[df.male == 0], color='red', label='female')
    plt.legend(loc=[1.1, 0.5])
    plt.ylabel('рост')
    plt.xlabel('масса')
plot(df_sample)

# Разделение данных
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_sample[['height', 'weight']], df_sample['male'], test_size=0.3, random_state=1)

# Нормировка данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=2023, solver='lbfgs').fit(X_train_scaled, y_train)

# Оценка модели
from sklearn.metrics import accuracy_score
y_pred = lr.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
