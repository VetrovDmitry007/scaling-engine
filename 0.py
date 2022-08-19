import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f'{arr=}')

eye = np.eye(4)
print(f'{eye=}')

# Преобразование массива в разряженную (сжатую) матрицу Scipy в формате CSR (сжатым хранением строкой)
csr_matrix = sparse.csr_matrix(eye)
print(f'\nРазряженная матрица формата CSR \n{csr_matrix}')


# Простой линейный график синусоидальной функции
# 1. Возвращает равномерно распределенные числа за указанный интервал.
x = np.linspace(-10, 10, 100)
# 2. Создаём второй масив с помощью синуса
y = np.sin(x)
# 3. Отрисовка графика
plt.plot(x,y)
# plt.show()


# Pandas

data = {'Name':['Иван', 'Алексей', 'Игорь', 'Сергей'],
        'Adder': ['Москва', 'Орёл', 'Курск', 'Белгород'],
        'Age': [22, 33, 27, 45]}

data_pandas = pd.DataFrame(data)
print(f'\n DataFrame \n{data_pandas}')

dt_1 = data_pandas[data_pandas.Age > 30]
print(f'\n Выборка из DataFrame \n{dt_1}')