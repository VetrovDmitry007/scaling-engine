# Изменение плотности в несбалансированных данных

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import mglearn
from mglearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Генерируем синтетические двумерные данные разной плотности
X, y = make_blobs(n_samples=450, centers=2, cluster_std=[7.0, 2.0], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)

# Диаграмма разброса данных
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.show()

report = classification_report(y_true=y_test, y_pred=svc.predict(X_test))
print(report)

# Изменение плотности
y_pred_lower = svc.decision_function(X_test) > -.8
report = classification_report(y_true=y_test, y_pred=y_pred_lower)
print(report)
