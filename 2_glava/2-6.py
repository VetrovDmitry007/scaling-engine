"""
Линейная классификация на наборе данных Breast Cancer
с использованием "Penalty" и методов "Логистическая регрессия"

Penalty влияет на регуляризацию и определяет,
будет ли модель использовать все доступные признаки или выберет лишь
подмножество признаков

L1 регуляризация, поскольку она ограничивает модель использованием лишь нескольких признаков
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
model_lg = LogisticRegression(max_iter=100_000).fit(X_train, y_train)
print(f'Оценка результата обучения (LogisticRegression max_iter=100_000): {model_lg.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression max_iter=100_000): {model_lg.score(X_test, y_test)}')

model_lg = LogisticRegression(max_iter=100_000, C=10).fit(X_train, y_train)
print(f'\nОценка результата обучения (LogisticRegression max_iter=100_000, C=10): {model_lg.score(X_train, y_train)}')
print(f'Оценка результата предсказания (LogisticRegression max_iter=100_000, C=10): {model_lg.score(X_test, y_test)}')