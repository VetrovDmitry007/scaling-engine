# Алгомеративная кластеризация на сентетическом наборе данных two_moons
#
from sklearn.datasets import make_moons, make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# генерируем синтетические данных two_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# X, y = make_blobs(random_state=170, cluster_std=[1.0, 2.5, 0.5],)

# Алгомеративная кластеризация
# clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=15).fit(X)
clustering = AgglomerativeClustering(n_clusters=2).fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], clustering.labels_)
plt.show()


