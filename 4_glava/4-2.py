# Биннинг
# преобразование непрерывного входного признака "fin" в набор данных "категориальный признак"

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('./data_dummies.csv', index_col=None, header=None, encoding = 'cp1251',
                   sep=';', names=['age', 'pol', 'fin', 'spec'])
print(data.head())

# print(data.describe())
# print(data.age.value_counts())
# print(data.age.max())
# print(data.age.min())
ages = data.age.values

# преобразование непрерывного входного признака набора данных в категориальный признак
bins = np.linspace(20, 60, 15)
which_bi = np.digitize(ages, bins=bins)
print(f'{bins=}')
print(f'{which_bi=}')

# увеличение размерности
which_bi = which_bi.reshape(-1,1)
# прямое кодирование категориального признака
encoder = OneHotEncoder(sparse=False).fit(which_bi)
X_binned = encoder.transform(which_bi)
# print(X_binned)

# (11, 4) + (11, 8) = (11, 12)
np_data = data.values
print(np_data.shape)
print(X_binned.shape)
np_data_new = np.hstack([np_data, X_binned])
print(np_data_new.shape)

data_binned = pd.DataFrame(np_data_new)
# data_binned = pd.DataFrame(np_data_new[:,[1,3,4]])
print(data_binned)


