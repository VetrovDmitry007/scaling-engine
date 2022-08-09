# Сокращения размерности данных алгоритмом "Анализ главных компонент (PCA)" на наборе данных "Breast Cancer"
# Применение логистической регрессии на уменьшеной размерности данных

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn
from sklearn.linear_model import LogisticRegression

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()

# Перед тем, как применить PCA, нужно отмасштабировать данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)

# оставляем первые две главные компоненты
pca = PCA(n_components=2)
# подгоняем модель PCA на наборе данных breast cancer
pca.fit(X_scaled)

# Преобразованием данных к первым двум главным компонентам
X_pca = pca.transform(X_scaled)
print("Форма исходного массива: {}".format(str(X_scaled.shape)))
print("Форма массива после сокращения размерности: {}".format(str(X_pca.shape)))

# Двумерная диаграмма рассеивания
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.show()

# Классификация 2-х мерных данных методом LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X_pca, cancer.target, random_state=0)
log_mod = LogisticRegression().fit(X_train, y_train)
print(f'Оценка результата обучения (LogisticRegression): {log_mod.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression): {log_mod.score(X_test, y_test)}')

