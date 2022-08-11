# Использование метрик для оценки отбора моделей
#    Использование "Перекрёстная 5-ти блочная проверка" с метрикой качества "AUC"
#    решении задачи «девять против остальных» для набора данных digits

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

digits = load_digits()
# digits.data.shape = (1797, 64)
# digits.target.shape = (1797, ), min = 0, max = 9
y = digits.target == 9
# y.shape = (1797, ), min = False, max = True
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

# Перекрёстная 5-ти блочная проверка с метрикой качества "Правильность"
scores_train = cross_val_score(SVC(), X_train, y_train, cv=5, scoring="accuracy")
scores_test = cross_val_score(SVC(), X_test, y_test, cv=5, scoring="accuracy")
print(f'Оценка результата обучения (SVC): {scores_train}')
print(f'Оценка результата предсказания (SVC): {scores_test.mean()}')

# "Перекрёстная 5-ти блочная проверка" с метрикой качества "AUC"
scores_train = cross_val_score(SVC(), X_train, y_train, cv=5, scoring="roc_auc")
scores_test = cross_val_score(SVC(), X_test, y_test, cv=5, scoring="roc_auc")
print(f'Оценка результата обучения (SVC -- AUC): {scores_train}')
print(f'Оценка результата предсказания (SVC -- AUC): {scores_test.mean()}')