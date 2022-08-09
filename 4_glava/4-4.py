# Полиномиальная регрессия на примере "Бостонского набора"
#
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures

boston = load_boston()
# boston = fetch_california_housing()

# 1.
# минимальных квадратов
X_train, X_test, y_train, y_test = train_test_split (boston.data, boston.target, random_state=0)
print(X_train.shape)
lr = LinearRegression().fit(X_train, y_train)
print(f'Оценка результата обучения (минимальных квадратов): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (минимальных квадратов): {lr.score(X_test, y_test)}')

# 2.
# масштабирование данных
X_train, X_test, y_train, y_test = train_test_split (boston.data, boston.target, random_state=0)
print(X_train.shape)
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
data = pd.DataFrame(X_train_scaler)
print(data.describe())

lr = Ridge(alpha=0.1).fit(X_train_scaler, y_train)
print(f'\nОценка результата обучения (Ridge): {lr.score(X_train_scaler, y_train)}')
print(f'Оценка результата предсказания (Ridge): {lr.score(X_test_scaler, y_test)}')

# 3.
# Создание полиномиальных и интерактивных признаков
poly = PolynomialFeatures(degree=4, include_bias=False).fit(X_train_scaler)
X_train_poly = poly.transform(X_train_scaler)
X_test_poly = poly.transform(X_test_scaler)
#
print(X_train_poly.shape)
# print(f'{poly.get_feature_names()=}')

lr = Ridge(alpha=0.1).fit(X_train_poly, y_train)
# lr = Lasso(alpha=0.01, max_iter=100_000).fit(X_train_poly, y_train)
print(f'\nОценка результата обучения (Ridge Polynomial): {lr.score(X_train_poly, y_train)}')
print(f'Оценка результата предсказания (Ridge Polynomial): {lr.score(X_test_poly, y_test)}')
