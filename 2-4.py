# Линейные модели регрессии на примере Бостонского набора
# 1. Метод наименьших квадратов
# 2. Метод "гребневая регрессия"
# 3. Метод Lasso

import numpy as np
from mglearn.datasets import load_extended_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# минимальных квадратов
lr = LinearRegression().fit(X_train, y_train)
print(f'Оценка результата обучения (минимальных квадратов): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (минимальных квадратов): {lr.score(X_test, y_test)}')

# гребневая регрессия
lr = Ridge(alpha=0.1).fit(X_train, y_train)
print(f'\nОценка результата обучения (гребневая регрессия): {lr.score(X_train, y_train)}')
print(f'Оценка результата предсказания (гребневая регрессия): {lr.score(X_test, y_test)}')

# Lasso. alpha -- степень сжатия коэфф. до нулевых значений, уменьшает кол-во признаков
lass = Lasso(alpha=0.01, max_iter=100_000).fit(X_train, y_train)
print(f'\nОценка результата обучения (Lasso): {lass.score(X_train, y_train)}')
print(f'Оценка результата предсказания (Lasso): {lass.score(X_test, y_test)}')
print(f'количество используемых признаков (Lasso): {np.sum(lass.coef_ != 0)}')