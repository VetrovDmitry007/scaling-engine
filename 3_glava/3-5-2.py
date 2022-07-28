# Кластеризация методом k-средних на сентетическом наборе данных
#
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn
import matplotlib.pyplot as plt

# Генерируем синтетические двумерные данные разной плотности
X, y = make_blobs(random_state=170, cluster_std=[1.0, 2.5, 0.5],)

# Кластеризация методом k-средних
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers=['o', '*', 'v'])
# plt.scatter(X[:, 0], X[:, 1], kmeans.labels_, cmap="jet", lw=3)
plt.show()


