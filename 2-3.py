# Линейная регрессия. y = a + b*x
# Метод нахождения сдвига "а" и коэффициента "b" -- "метод наименьших квадратов"
# "метод наименьших квадратов" -- https://td.chem.msu.ru/uploads/files/courses/special/expmethods/statexp/LabLecture03.pdf

from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Синтетический набор данных "make_wave"
# X.shape = (60,1), y.shape = - (60,)
X, y = make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
print(f'Сдвиг модели : {lr.intercept_=}')
print(f'Коэффициент модели: {lr.coef_=}')
print(f'Результат предсказания: {lr.predict(X_test)}')
print(f'Оценка результата предсказания: {lr.score(X_test, y_test)}')