# Ядерный метод опорных векторов
# используется прималдом кол-ве признаков, требуется нормализация данных

# from mglearn.make_blobs import make_blobs
from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np

# Генерация синтетического набора данных
X, y = make_blobs(centers=5, n_features=2, random_state=5)
# X, y = make_blobs(centers=4, n_features=2)

y = y % 2
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Признак 0")
# plt.ylabel("Признак 1")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
# model = LinearSVC(random_state=0).fit(X_train, y_train)

# Нормализация данных
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_norm = (X_train - min_on_training) / range_on_training
X_test_norm = (X_test - min_on_training) / range_on_training
model = SVC(kernel='rbf', C=10, gamma=1, random_state=0).fit(X_train_norm, y_train)
print(f'Оценка результата обучения (SVC): {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (SVC): {model.score(X_test_norm, y_test)}')

# mglearn.plots.plot_2d_separator(model, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.show()