# Поиск оптимальных значений ключевых параметров модели -- "Простой решетчатый поиск"
# на примере синтетического набора данных

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=700, random_state=170, centers=2, cluster_std=[1.0, 2.5])
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=0)
print(f'{X_train_val.shape=}')
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=0)
print(f'{X_train.shape=}')
print(f'{X_val.shape=}')

best_score = 0

# "Простой решетчатый поиск"
for c in [0.001, 0.01, 0.1, 1, 10, 100]:
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        mod_svc = SVC(C=c, gamma=gamma, random_state=0)
        mod_svc.fit(X_train, y_train)
        score = mod_svc.score(X_val, y_val)
        # если получаем наилучшее значение правильности, сохраняем значение и параметры
        if score > best_score:
            best_score = score
            best_parameters = {'C': c, 'gamma': gamma}

print(f'{best_score=}')
print(f'{best_parameters=}')

model = SVC(**best_parameters).fit(X_train, y_train)
print(f'\nОценка результата прогноза (SVC): {model.score(X_test, y_test)}')