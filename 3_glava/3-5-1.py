# Кластеризация методом k-средних набора данных "Breast Cancer"
#
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cancer = load_breast_cancer()
# Масштабирование данных
X = StandardScaler().fit_transform(cancer.data)

# Кластеризация методом k-средних
kmeans = KMeans(n_clusters=2, random_state=1).fit(X)

# Сравнение факта с прогнозом
def get_prc(arr_1, arr_2):
    cn_all = arr_1.shape[0]
    cn_false = np.sum(arr_2 != arr_1)
    prc = (cn_false * 100) /  cn_all
    print(f'Вероятность прогноза: { round(prc, 4)} %')

get_prc(cancer.target, kmeans.labels_)
