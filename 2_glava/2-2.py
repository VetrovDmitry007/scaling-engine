# Регрессия
# Использование алгоритма "k ближайших соседей" на синтетическом наборе "make_wave"

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from mglearn.datasets import make_wave
from mglearn.plots import plot_knn_regression
import matplotlib.pyplot as plt

# X.shape = (40,1), y.shape = - (40,)
X, y = make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
print(f'Результат предсказания: {knn.predict(X_test)}')
# knn.score() => R2 - коэффициент детерминации. R2 приним. знач. от 0 до 1
print(f'Оценка результата предсказания: {knn.score(X_test, y_test)}')

# Визуализация применения алгоритма "k ближайших соседей" для регрессии
plot_knn_regression(n_neighbors=3)
plt.show()