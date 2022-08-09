# 1. Использование класса -- GridSearchCV
# Поиск оптимальных значений ключевых параметров модели
# -- "Простой решетчатый поиск" + "Перекрёстная 5-ти блочная проверка" на примере синтетического набора данных
# 2. Экономичный решётчатый поиск
# 3. Визуализация при помощи тепловой карты

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=700, random_state=170, centers=2, cluster_std=[1.0, 2.5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Параметры модели
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# 2. Экономичный решётчатый поиск
# param_grid = [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
#               {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

# Класс GridSearchCV  --"Простой решетчатый поиск" + "Перекрёстная 5-ти блочная проверка"
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Найденная комбинация лучших параметров: {grid_search.best_params_}')
print(f'Наилучшее значение правильности перекрестной проверки: {grid_search.best_score_}')

print(f'\nОценка результата обучения GridSearchCV(SVC): {grid_search.score(X_train, y_train)}')
print(f'Оценка результата предсказания GridSearchCV(SVC): {grid_search.score(X_test, y_test)}')

# 3. Визуализация при помощи тепловой карты
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(6, 6)
# строим теплокарту средних значений правильности перекрестной проверки
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'], ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
plt.show()