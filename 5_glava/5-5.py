# 1. Вложенная перекрёстная проверка -- cross_val_score( GridSearchCV(..) ..)
# 2. Распараллеливание задачи

import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, ShuffleSplit, \
    LeaveOneOut, KFold
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=700, random_state=170, centers=2, cluster_std=[1.0, 2.5])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

param_grid = [{'kernel': ['rbf'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]

# score = cross_val_score( GridSearchCV(SVC(), param_grid, cv=5).fit(X_train, y_train), X_test, y_test, cv=4)
# print(score.mean())

# Cтратифицированная k-блочная перекрестная проверка
gn =  KFold(random_state=0, n_splits=4, shuffle=True)

t1 = time.perf_counter()
score = cross_val_score( GridSearchCV(SVC(), param_grid, cv=5).fit(X_train, y_train), X_test, y_test, cv=gn)
print(f'Оценка результата прогноза (SVC): {score.mean()}')
print(f'Time: {time.perf_counter() - t1}\n')

# Распараллеливание задачи. n_jobs=-1 -- использовать все ядра
t1 = time.perf_counter()
score = cross_val_score( GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1).fit(X_train, y_train), X_test, y_test, cv=gn, n_jobs=-1)
print(f'Оценка результата прогноза (SVC): {score.mean()}')
print(f'Time: {time.perf_counter() - t1}')



