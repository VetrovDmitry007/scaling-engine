# Деревья решений
# Важность признаков в деревьях
# Классификация на наборе данных "Breast Cancer"
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

# Визуализация "Деревьев решений"
# mglearn.plots.plot_tree_progressive()
# plt.show()

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

model_tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print(f'Оценка результата обучения (DecisionTreeClassifier): {model_tree.score(X_train, y_train)}')
print(f'Оценка результата предсказания (DecisionTreeClassifier): {model_tree.score(X_test, y_test)}')

model_tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
print(f'\nОценка результата обучения (DecisionTreeClassifier, max_depth=4): {model_tree.score(X_train, y_train)}')
print(f'Оценка результата предсказания (DecisionTreeClassifier, max_depth=4): {model_tree.score(X_test, y_test)}')


# Важность признаков в деревьях
# 1.
# print(f'\nВажность признаков в деревьях:\n{model_tree.feature_importances_=}\n')
# 2.
# for name, score in zip(cancer["feature_names"], model_tree.feature_importances_):
#  print(name, score)
# 3.
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.show()
plot_feature_importances_cancer(model_tree)


# Взаимосвять глубины дерева и правильности предсказания
ls_train = []
ls_test = []
for n in range(1, 7):
    model = DecisionTreeClassifier(max_depth=n, random_state=0).fit(X_train, y_train)
    ls_train.append(model.score(X_train, y_train))
    ls_test.append(model.score(X_test, y_test))

plt.semilogx(range(1, 7), ls_train, label='Обучающие данные')
plt.semilogx(range(1, 7), ls_test, label='Тестовые данные')
plt.xlabel("Глубина")
plt.ylabel("Правильность")
plt.legend()
# plt.show()
