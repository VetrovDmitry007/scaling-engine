# Уменьшение признаков методом "Одномерной статистики" в логистической регрессии
# на наборе данных "Breast Cancer"

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
print(f"{X_train.shape=}")

select = SelectPercentile(percentile=40).fit(X_train, y_train)
# преобразовываем данные
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
print(f"{X_train_selected.shape=}")

lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train, y_train)
print(f'\nОценка результата обучения (LogisticRegression): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression): {lr.score(X_test, y_test)}')

lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_selected, y_train)
print(f'\nОценка результата обучения (LogisticRegression - SelectPercentile): {lr.score(X_train_selected, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression - SelectPercentile): {lr.score(X_test_selected, y_test)}')

