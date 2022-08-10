# Кривые точности-полноты и ROC-кривые

# 1. Функция precision_recall_curve возвращает список значений точности и полноты для всех возможных пороговых значений (всех
#   значений решающей функции) в отсортированном виде

# 2. Вычисление среднего значения кривой точности-полноты
#  методом "средней точности" -- average_precision_score

from sklearn.model_selection import train_test_split
from mglearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=4500, centers=2, cluster_std=[7.0, 2.0], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
print(f"Правильность svc: {svc.score(X_test, y_test)}")

# используем больший объем данных, чтобы получить более гладкую кривую
# precision_recall_curve -- возвращает список значений точности и полноты
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

# Находим ближайший к нулю порог
close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
label="порог 0", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="кривая точности-полноты")
plt.xlabel("Точность")
plt.ylabel("Полнота")
plt.legend(loc="best")


# 2. Вычисление среднего значения кривой точности-полноты
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print(f"Средняя точность svc: {ap_svc}")

plt.show()