# 1. Использование "Генаратора разбиений" к перекрёстной 5-ти блочной проверки на примере синтетического набора
# 2. Cтратифицированная k-блочная перекрестная проверка
#    с процентным сохранением образцов для каждого класса
# 3. Перекрёстная проверка с исключением по одному
# 4. Перекрестная проверка со случайными перестановками при разбиении
#
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score

# Генерируем синтетические двумерные данные разной плотности
# X, y = make_blobs(random_state=170, cluster_std=[1.0, 2.5, 0.5])
X, y = make_blobs(n_samples=100, random_state=170, centers=2, cluster_std=[1.0, 2.5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# data = pd.DataFrame(X_train_scaled)
# print(data.describe())

# Двумерная диаграмма рассеивания
mglearn.discrete_scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train)
plt.show()

lr = LogisticRegression(max_iter=100_000, C=1).fit(X_train_scaled, y_train)

# 1. Генератор разбиений
# gn = KFold(random_state=0, n_splits=4, shuffle=True)

# 2. Cтратифицированная k-блочная перекрестная проверка
gn = StratifiedKFold(random_state=0, n_splits=5, shuffle=True)

# 3. Перекрёстная проверка с исключением по одному
# gn = LeaveOneOut()

# 4. Перекрестная проверка со случайными перестановками при разбиении
# gn = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)

# Перекрёстная 5-ти блочная проверка
scores_train = cross_val_score(lr, X_train_scaled, y_train, cv=gn)
scores_test = cross_val_score(lr, X_test_scaled, y_test, cv=gn)
print(f'Оценка результата обучения (LogisticRegression): {scores_train}')
print(f'Оценка результата предсказания (LogisticRegression): {scores_test}')
print(f'Оценка результата предсказания (LogisticRegression): {scores_test.mean()}')


