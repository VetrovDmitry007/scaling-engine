"""
Пример использования алгоритмов ML к набору данных diabetes,
"""
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

ds_diabetes = load_diabetes()
X = ds_diabetes.data
y = ds_diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# """
ml_model = LinearRegression().fit(X_train, y_train)
print(f'Оценка результата обучения (LinearRegression): {ml_model.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LinearRegression): {ml_model.score(X_test, y_test)}')

ml_model = Ridge(alpha=0.1).fit(X_train, y_train)
print(f'\nОценка результата обучения (Ridge): {ml_model.score(X_train, y_train)}')
print(f'Оценка результата предсказания (Ridge): {ml_model.score(X_test, y_test)}')

ml_model = Lasso(alpha=0.01, max_iter=100_000).fit(X_train, y_train)
print(f'\nОценка результата обучения (Lasso): {ml_model.score(X_train, y_train)}')
print(f'Оценка результата предсказания (Lasso): {ml_model.score(X_test, y_test)}')

tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_train, y_train)
print(f'\nОценка результата обучения (DecisionTreeRegressor): {tree.score(X_train, y_train)}')
print(f'Оценка результата предсказания (DecisionTreeRegressor): {tree.score(X_test, y_test)}')

model_rand_forest = RandomForestRegressor(max_depth=3, random_state=0, max_features=8).fit(X_train, y_train)
print(f'\nОценка результата обучения (RandomForestRegressor): {model_rand_forest.score(X_train, y_train)}')
print(f'Оценка результата предсказания (RandomForestRegressor): {model_rand_forest.score(X_test, y_test)}')

model_forest = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
print(f'\nОценка результата обучения (GradientBoostingRegressor): {model_forest.score(X_train, y_train)}')
print(f'Оценка результата предсказания (GradientBoostingRegressor): {model_forest.score(X_test, y_test)}')

from sklearn import tree
dtree = tree.DecisionTreeRegressor(min_samples_split=20).fit(X_train, y_train)
print(f'\nОценка результата обучения (DecisionTreeRegressor): {dtree.score(X_train, y_train)}')
print(f'Оценка результата предсказания (DecisionTreeRegressor): {dtree.score(X_test, y_test)}')

# Нормализация данных
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_norm = (X_train - min_on_training) / range_on_training
X_test_norm = (X_test - min_on_training) / range_on_training

model = MLPRegressor(max_iter=100_000, random_state=0).fit(X_train_norm, y_train)
print(f'\nОценка результата обучения (MLP), с нормализацией данных: {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (MLP), с нормализацией данных: {model.score(X_test_norm, y_test)}')

# """

# df_diabetes = pd.DataFrame(X, columns=ds_diabetes.feature_names)
# print(df_diabetes.describe())
# df_diabetes['target'] = ds_diabetes.target
# df_diabetes['std_data'] = list(df_diabetes.std(axis=1))
# print(df_diabetes)
# print(df_diabetes.count(axis=0))

