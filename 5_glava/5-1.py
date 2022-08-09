# Перекрёстная 5-ти блочная проверка на примере синтетического набора
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score

# Генерируем синтетические двумерные данные разной плотности
# X, y = make_blobs(random_state=170, cluster_std=[1.0, 2.5, 0.5])
X, y = make_blobs(random_state=170, centers=2, cluster_std=[1.0, 2.5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# data = pd.DataFrame(X_train_scaled)
# print(data.describe())

# Двумерная диаграмма рассеивания
# mglearn.discrete_scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train)
# plt.show()

lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_scaled, y_train)
# Перекрёстная 5-ти блочная проверка
scores_train = cross_val_score(lr, X_train_scaled, y_train, cv=5)
scores_test = cross_val_score(lr, X_test_scaled, y_test, cv=5)
print(f'Оценка результата обучения (LogisticRegression): {scores_train}')
print(f'Оценка результата предсказания (LogisticRegression): {scores_test.mean()}')