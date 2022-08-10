# Оценка качества бинарной классификации в несбалансированном наборе данных
# 1. при помощи метрики "Матрицы ошибок"
# 2. F-мера -- гармоническое среднее точности и полноты
# 3. Отчет о точности полноте и f1-мере

from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

digits = load_digits()
# digits.data.shape = (1797, 64)
# digits.target.shape = (1797, ), min = 0, max = 9
y = digits.target == 9
# y.shape = (1797, ), min = False, max = True
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

tree = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
confusion = confusion_matrix(y_test, pred_tree)
print(f'Прогноз DecisionTreeClassifier:{tree.score(X_test, y_test)}')
print(f'Матрица ошибок DecisionTreeClassifier: {confusion}')

print(f'Правильность DecisionTreeClassifier: {(390+23)/(390+23+13+24)=}')
# [[390  13]
# [ 24  23]]
# 0.9177777777777778
print(f'F-мера -- гармоническое среднее точности и полноты DecisionTreeClassifier: {f1_score(y_test, pred_tree)}')

lg = LogisticRegression(max_iter=500).fit(X_train, y_train)
pred_logreg = lg.predict(X_test)
confusion = confusion_matrix(y_test, pred_logreg)
print(f'\nПрогноз LogisticRegression:{lg.score(X_test, y_test)}')
print(f'Матрица ошибок LogisticRegression: {confusion}')
print(f'F-мера -- гармоническое среднее точности и полноты LogisticRegression: {f1_score(y_test, pred_logreg)}')

print(f'\nОтчет о точности полноте и f1-мере: \n{classification_report(y_true=y_test, y_pred=pred_logreg, target_names=["not nine","nine"])}')
