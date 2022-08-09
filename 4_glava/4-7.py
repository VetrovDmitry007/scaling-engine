# Уменьшение признаков методом "Итеративный отбор признаков" в логистической регрессии
# на наборе данных "Breast Cancer"

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# Итеративный отбор признаков
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=25)
select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

print(X_train_rfe.shape)

# Логистическая регрессия
lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train, y_train)
print(f'\nОценка результата обучения (LogisticRegression): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression): {lr.score(X_test, y_test)}')

# Логистическая регрессия + Итеративный отбор признаков. (426, 30) -> (426, 25)
lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_rfe, y_train)
print(f'\nОценка результата обучения (LogisticRegression - RFE): {lr.score(X_train_rfe, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression - RFE): {lr.score(X_test_rfe, y_test)}')

# Масштабирование данных
scaler = StandardScaler().fit(X_train_rfe)
X_train_scaler = scaler.transform(X_train_rfe)
X_test_scaler = scaler.transform(X_test_rfe)
# print(pd.DataFrame(X_train_scaler).describe())

# Логистическая регрессия + Итеративный отбор признаков + Масштабирование данных
lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_scaler, y_train)
print(f'\nОценка результата обучения (LogisticRegression - RFE - StandardScaler): {lr.score(X_train_scaler, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression - RFE - StandardScaler): {lr.score(X_test_scaler, y_test)}')