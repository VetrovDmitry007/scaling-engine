# Нейронная модель MLP -- многослойный перцептрон
# Классификация на наборе данных "Breast Cancer"
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

# X=(569, 30), y=(569,)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# Нормализация данных
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_norm = (X_train - min_on_training) / range_on_training
X_test_norm = (X_test - min_on_training) / range_on_training

model = MLPClassifier(max_iter=1000, random_state=0).fit(X_train_norm, y_train)
print(f'Оценка результата обучения (MLP), с нормализацией данных: {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (MLP), с нормализацией данных: {model.score(X_test_norm, y_test)}')

model = MLPClassifier(solver='lbfgs', activation='tanh',
                        random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train_norm, y_train)
print(f'\nОценка результата обучения (MLP), с нормализацией данных: {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (MLP), с нормализацией данных: {model.score(X_test_norm, y_test)}')

# adam — один из самых эффективных алгоритмов оптимизации в обучении нейронных сетей
# alpha -- скорость обучения
model = MLPClassifier(max_iter=1000, solver='adam', activation='tanh',
                        random_state=0, hidden_layer_sizes=[10,10], alpha=1).fit(X_train_norm, y_train)
print(f'\nОценка результата обучения (MLP), с нормализацией данных: {model.score(X_train_norm, y_train)}')
print(f'Оценка результата предсказания (MLP), с нормализацией данных: {model.score(X_test_norm, y_test)}')





