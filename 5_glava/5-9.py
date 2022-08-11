# Метрики для мультиклассовой классификации
# 1. Матрица ошибок
# 2. Мультиклассовый вариант F-меры (гармоническое среднее точности и полноты)

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# digits.data.shape = (1797, 64)
# digits.target.shape = (1797, ), min = 0, max = 9
digits = load_digits()
# param_grid = [{'solver':['liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
#               {'solver':['lbfgs'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
#               {'solver':['sag'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
#               {'solver':['saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
#               {'solver':['newton-cg'], 'C': [0.001, 0.01, 0.1, 1, 10, 100]},
#               ]

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

# Масштабирование
scaler = StandardScaler()
# scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

lr = LogisticRegression(C= 0.1, solver= 'saga').fit(X_train_scaler, y_train)
pred = lr.predict(X_test_scaler)

data = pd.DataFrame(X_test_scaler)
print(data.describe())


# Класс GridSearchCV  --"Простой решетчатый поиск" + "Перекрёстная 5-ти блочная проверка"
# grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid_search.fit(X_train, y_train)
# pred = grid_search.predict(X_test)
# print(f'Найденная комбинация лучших параметров: {grid_search.best_params_}')
# print(f'Наилучшее значение правильности перекрестной проверки: {grid_search.best_score_}')

print(f'\nОценка результата обучения GridSearchCV(LogisticRegression): {lr.score(X_train_scaler, y_train)}')
print(f'Оценка результата предсказания GridSearchCV(LogisticRegression): {lr.score(X_test_scaler, y_test)}')

print(f"\nТочность: {accuracy_score(y_test, pred)}")
print(f"Матрица ошибок:\n {confusion_matrix(y_test, pred)}")

print(f'\nОтчет о точности полноте и f1-мере\n{classification_report(y_test, pred)}')

# Мультиклассовый вариант f-меры
print(f'F-мера -- Мультиклассовое гармоническое среднее точности и полноты: {f1_score(y_test, pred, average="macro")}')
