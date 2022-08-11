# Использование метрик для оценки отбора моделей
#    Использование "Простой решетчатый поиск" с метрикой качества "AUC"
#    решении задачи «девять против остальных» для набора данных digits

from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

digits = load_digits()
# digits.data.shape = (1797, 64)
# digits.target.shape = (1797, ), min = 0, max = 9
y = digits.target == 9
# y.shape = (1797, ), min = False, max = True
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# Простой решетчатый поиск с метрикой качества "Правильность"
grid_search = GridSearchCV(SVC(), param_grid, scoring="accuracy")
grid_search.fit(X_train, y_train)
print(f'Найденная комбинация лучших параметров: {grid_search.best_params_}')
print(f'Наилучшее значение правильности перекрестной проверки: {grid_search.best_score_}')
print(f'Правильность на тестовом наборе GridSearchCV(SVC): {grid_search.score(X_test, y_test)}')
print(f'AUC на тестовом наборе GridSearchCV(SVC) {roc_auc_score(y_test, grid_search.decision_function(X_test))}')

# Простой решетчатый поиск с метрикой качества "AUC"
grid_search = GridSearchCV(SVC(), param_grid, scoring="roc_auc")
grid_search.fit(X_train, y_train)
print(f'\nНайденная комбинация лучших параметров (SVC -- AUC): {grid_search.best_params_}')
print(f'Наилучшее значение правильности перекрестной проверки (SVC -- AUC): {grid_search.best_score_}')
print(f'Правильность на тестовом наборе GridSearchCV(SVC -- AUC): {grid_search.score(X_test, y_test)}')
print(f'AUC на тестовом наборе GridSearchCV(SVC -- AUC) {roc_auc_score(y_test, grid_search.decision_function(X_test))}')