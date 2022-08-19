"""
Анализ тональности на наборе постов из Твиттера
при помощи модели "мешок слов"
"""
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import no_grad

data_negat = pd.read_csv(r'D:\Python_2\ML\DataSet\negative.csv', index_col=None, header=None, sep=';')
data_posit = pd.read_csv(r'D:\Python_2\ML\DataSet\positive.csv', index_col=None, header=None, sep=';')
# Формирования X_data, y_data
arr_negat_words = data_negat.values[:, 3]
arr_negat_y = np.zeros((arr_negat_words.shape)).astype(int)
arr_posit_words = data_posit.values[:, 3]
arr_posit_y = np.ones((arr_posit_words.shape)).astype(int)

X_data = np.concatenate((arr_negat_words, arr_posit_words), axis=0)
y_data = np.concatenate((arr_negat_y, arr_posit_y), axis=0)
X_text_train, X_text_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

cn = 1
nlp = spacy.load("ru_core_news_sm")

def custom_tokenizer(document):
    global cn
    global nlp
    print(f'{cn=}')
    cn += 1
    # doc_spacy = nlp(document, entity=False, parse=False)
    doc_spacy = nlp(document)
    return [token.lemma_ for token in doc_spacy if token.dep_ not in ['punct', 'case']]

# CountVectorizer() -- модель "мешок слов"
# Конвертирование текста в матрицу векторов (предложение --> вектор)
vect = CountVectorizer(tokenizer = custom_tokenizer).fit(X_text_train)
# vect = CountVectorizer(ngram_range=(1,3), min_df=2).fit(X_text_train)
X_train = vect.transform(X_text_train)
X_test = vect.transform(X_text_test)
print(f'{X_train.shape=}, {y_train.shape=}')
print(f'{X_test.shape=}, {y_test.shape=}')

print(vect.vocabulary_)
exit(0)

# scores = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
# print(f"Правильность перекр. проверки: {scores}")
# print(f"Средняя правильность перекр. проверки: {np.mean(scores)}")

# Уменьшение признаков методом "Одномерной статистики"
select = SelectPercentile(percentile=60).fit(X_train, y_train)
# преобразовываем данные
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)
print(f'{X_train_selected.shape=}, {y_train.shape=}')
print(f'{X_test_selected.shape=}, {y_train.shape=}')

# pipe = Pipeline([('select', SelectPercentile()), ('lr', LogisticRegression())])
# param_grid = {'select__percentile':[10, 20, 30, 40, 50, 60, 70, 80],
#               'lr__C': [0.001, 0.01, 0.1, 1, 10],
#               'lr__max_iter': [1000]}
# param_grid = {'select__percentile':[60], 'lr__C': [1], 'lr__max_iter': [1000]}
# grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)
# grid.fit(X_train, y_train)
# print(f"Лучшие параметры: {grid.best_params_}")
# print(f"Правильность : {grid.score(X_test, y_test)}")
"""
Лучшие параметры: {'lr__C': 1, 'select__percentile': 60}
Правильность прогноза: 0.75
"""

lr = LogisticRegression(max_iter=1000, C=1)
lr.fit(X_train_selected, y_train)
print(f'\nПравильность обучения: {lr.score(X_train_selected, y_train)}')
print(f'Правильность прогноза: {lr.score(X_test_selected, y_test)}')
"""
Правильность обучения: 0.9141631153563556
Правильность прогноза: 0.7702657426510783
"""

# model_rand_forest = RandomForestClassifier(n_estimators=100, max_features=100, random_state=0).fit(X_train_selected, y_train)
# print(f'\nОценка результата обучения (RandomForestClassifier, n_estimators=100, max_features=30): {model_rand_forest.score(X_train_selected, y_train)}')
# print(f'Оценка результата предсказания (RandomForestClassifier, n_estimators=100, max_features=30): {model_rand_forest.score(X_test_selected, y_test)}')
"""
Оценка результата обучения (RandomForestClassifier, n_estimators=100, max_features=100): 0.9920587803085966
Оценка результата предсказания (RandomForestClassifier, n_estimators=100, max_features=100): 0.7163942231391842
"""

