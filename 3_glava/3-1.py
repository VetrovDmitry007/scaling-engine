# Примеры преобразования (масштабирования данных) данных
# MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.svm import SVC

mglearn.plots.plot_scaling()
plt.show()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
df_cancer = pd.DataFrame(X_train)
# print(df_cancer.describe())

scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax = scaler.transform(X_test)

df_cancer = pd.DataFrame(X_train_minmax)
# print(df_cancer.describe())

model = SVC(kernel='rbf', C=10, gamma=1, random_state=0).fit(X_train, y_train)
print(f'Оценка результата обучения (SVC): {model.score(X_train, y_train)}')
print(f'Оценка результата предсказания (SVC): {model.score(X_test, y_test)}')

model = SVC(kernel='rbf', C=10, gamma=1, random_state=0).fit(X_train_minmax, y_train)
print(f'\nОценка результата обучения (SVC): {model.score(X_train_minmax, y_train)}')
print(f'Оценка результата предсказания (SVC): {model.score(X_test_minmax, y_test)}')

