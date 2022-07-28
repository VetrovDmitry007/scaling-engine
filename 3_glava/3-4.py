# Выделение признаков алгоритмом "Факторизация неотрицательных матриц (NMF)"
# в наборе данных "Breast Cancer"

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVC

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# Перед тем, как применить PCA, нужно отмасштабировать данные
scaler = Normalizer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Факторизация неотрицательных матриц
pca = NMF(n_components=10, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model = SVC(kernel='rbf', C=10, gamma=0.1, random_state=0).fit(X_train_pca, y_train)

print(f'Оценка результата обучения (SVC): {model.score(X_train_pca, y_train)}')
print(f'Оценка результата предсказания (SVC): {model.score(X_test_pca, y_test)}')