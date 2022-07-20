# Ядерный метод опорных векторов
# Классификация на наборе данных Breast Cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

model = SVC(kernel='rbf', C=10, gamma=1, random_state=0).fit(X_train, y_train)
print(f'Оценка результата обучения (SVC): {model.score(X_train, y_train)}')
print(f'Оценка результата предсказания (SVC): {model.score(X_test, y_test)}')

# Нормализация данных
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_norm = (X_train - min_on_training) / range_on_training
X_test_norm = (X_test - min_on_training) / range_on_training

model = SVC(kernel='rbf', C=10, gamma=1, random_state=0).fit(X_train_norm, y_train)
print(f'\nОценка результата обучения (SVC), с нормализацией данных: {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (SVC), с нормализацией данных: {model.score(X_test_norm, y_test)}')


