# DBSCAN – плотностный алгоритм кластеризации пространственных данных с присутствием шума
# на сентетическом наборе данных two_moons / make_blobs

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mglearn
import matplotlib.pyplot as plt

# генерируем синтетические данных
X, y = make_moons(n_samples=200, noise=0.05, random_state=30)
# X, y = make_blobs(random_state=170, cluster_std=[1.0, 2.5, 0.5],)

X = StandardScaler().fit_transform(X)
clusters = DBSCAN(min_samples=8, eps=0.37).fit_predict(X)

# X = MinMaxScaler().fit_transform(X)
# clusters = DBSCAN(min_samples=10, eps=0.15).fit_predict(X)

# Двумерная диаграмма рассеивания
mglearn.discrete_scatter(X[:, 0], X[:, 1], clusters)
# plt.scatter(X[:, 0], X[:, 1], c=clusters, s=60)
plt.show()