from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

iris_datase = load_iris()
print(f'Iris_datase type:{type(iris_datase)}')
print(iris_datase.keys())
print(f'{iris_datase["target_names"]=}')
print(f"\nРазмерность элемента target: {iris_datase['target'].shape}")
print(f"Размерность элемента data: {iris_datase['data'].shape}")
print(iris_datase['data'][:6])

# Разделяем на обучающий и тестовый набор данных
X_train, X_test, y_train, y_test = train_test_split(iris_datase['data'], iris_datase['target'], random_state=0)

print(f'{X_train.shape=}')
print(f'{y_train.shape=}')
print(f'{X_test.shape=}')
print(f'{y_test.shape=}')

# Исследование данных через их визуализацию
iris_dataframe = pd.DataFrame(X_train, columns=iris_datase['feature_names'])
# print(iris_dataframe.head)
# Отрисовка матрицы графиков разброса
grr = pd.plotting.scatter_matrix(iris_dataframe, marker='o', c = y_train, hist_kwds={'bins':20})
# plt.show()

# Использование классификатора на основе модели "k ближайших соседей"
# neighbors -- соседи
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# Обучение модели
knn.fit(X_train, y_train)

# Получения прогноза
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(f'Спрогнозированая метка: {prediction=}')

# Оценка качества модели
y_pred = knn.predict(X_test)
# через np.mean - среднее арифметическое, находим долю правельных предсказаний
print(f'Правильность на тестовом наборе v1: {np.mean(y_pred == y_test)=}')
print(f'Правильность на тестовом наборе v2: {knn.score(X_test, y_test)=}')
# print(y_pred)

