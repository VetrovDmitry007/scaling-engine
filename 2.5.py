"""
Линейный модели классификации -- это классификатор разделяющий два класса с помошью линии, проскости, гиперплоскости
Два наиболее распространённых метода:
1. Логистическая регрессия -- LogisticRegression
2. Метод опорных векторов -- LinearSVC
"""
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mglearn.datasets import make_forge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import mglearn


# X.shape=(26, 2),  y.shape=(26,)
from mglearn import discrete_scatter

X, y = make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Исследование данных через их визуализацию
discrete_scatter(X[:, 0], X[:, 1], y)
# plt.show()

model_lg = LogisticRegression(max_iter=100_000, C=1).fit(X_train, y_train)
print(f'Предсказание (LogisticRegression): {model_lg.predict(X_test)}')
print(f'Оценка результата обучения (LogisticRegression): {model_lg.score(X_test, y_test)}')
print(f'Оценка результата предсказания (LogisticRegression): {model_lg.score(X_test, y_test)}')


model_svc = LinearSVC(max_iter=1000_000, C=100).fit(X_train, y_train)
print(f'\nПредсказание (LinearSVC): {model_svc.predict(X_test)}')
print(f'Оценка результата обучения (LinearSVC): {model_svc.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LinearSVC): {model_svc.score(X_test, y_test)}')

# C = min -- подстроиться под большинство точек
# C = max -- цель - классифицировать каждую точку
mglearn.plots.plot_linear_svc_regularization()
plt.show()