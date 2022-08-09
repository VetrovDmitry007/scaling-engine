# Биннинг и Взаимодействие признаков
#
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('./data_dummies.csv', index_col=None, header=None, encoding = 'cp1251',
                   sep=';', names=['age', 'pol', 'fin', 'spec'])

# преобразование непрерывного входного признака набора данных в категориальный признак
bins = np.linspace(20, 60, 15)
ages = data.age.values
which_bi = np.digitize(ages, bins=bins)

# увеличение размерности
which_bi = which_bi.reshape(-1,1)
# прямое кодирование категориального признака
encoder = OneHotEncoder(sparse=False).fit(which_bi)
X_binned = encoder.transform(which_bi)
print(X_binned.shape)

X_combined = np.hstack([X_binned, ages.reshape(-1,1) * X_binned])
print(X_combined.shape)
print(X_combined)