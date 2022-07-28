# Двухклассовая классификации
# Использование алгоритма "k ближайших соседей" на синтетическом наборе "make_forge"

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from mglearn.datasets import make_forge
from mglearn import discrete_scatter
from mglearn.plots import plot_knn_classification

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Синтетический набор для двухклассовой классификации
X, y = make_forge()
print(f'{X.shape=}')
print(f'{y.shape=}')

# Исследование данных через их визуализацию
# Диаграмма рассеяности для набора данных
# Вариант - 1
# forge_dataframe = pd.DataFrame(X, columns=['Первый признак', 'Второй признак'])
# grr = pd.plotting.scatter_matrix(forge_dataframe, marker='o', c = y, hist_kwds={'bins':20})
# plt.show()
# Вариант - 2
discrete_scatter(X[:, 0], X[:, 1], y)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


knn = KNeighborsClassifier(n_neighbors=1)
# Обучение модели
knn.fit(X_train, y_train)

print(f'Правильность на тестовом наборе v2: {knn.score(X_test, y_test)=}')

# Визуализация алгоритма "k ближайших соседей"
plot_knn_classification(n_neighbors=3)
plt.show()