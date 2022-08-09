# Уменьшение признаков методом "Отбор признаков на основе модели" в логистической регрессии
# на наборе данных "Breast Cancer"

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# Итеративный отбор признаков
# select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="0.35 * mean")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
X_test_l1 = select.transform(X_test)

print(X_train_l1.shape)

# Логистическая регрессия
lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train, y_train)
print(f'\nОценка результата обучения (LogisticRegression): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression): {lr.score(X_test, y_test)}')

# Логистическая регрессия + Итеративный отбор признаков. (426, 30) -> (426, 25)
lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_l1, y_train)
print(f'\nОценка результата обучения (LogisticRegression - SelectFromModel): {lr.score(X_train_l1, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression - SelectFromModel): {lr.score(X_test_l1, y_test)}')

