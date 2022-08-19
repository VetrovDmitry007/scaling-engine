"""
Пример использования алгоритмов ML к набору данных diabetes,
"""
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile
import pickle


# """
# ml_model = LinearRegression().fit(X_train, y_train)
# print(f'Оценка результата обучения (LinearRegression): {ml_model.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (LinearRegression): {ml_model.score(X_test, y_test)}')
#
# ml_model = Ridge(alpha=0.1).fit(X_train, y_train)
# print(f'\nОценка результата обучения (Ridge): {ml_model.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (Ridge): {ml_model.score(X_test, y_test)}')
#
# ml_model = Lasso(alpha=0.01, max_iter=100_000).fit(X_train, y_train)
# print(f'\nОценка результата обучения (Lasso): {ml_model.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (Lasso): {ml_model.score(X_test, y_test)}')
#
# tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_train, y_train)
# print(f'\nОценка результата обучения (DecisionTreeRegressor): {tree.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (DecisionTreeRegressor): {tree.score(X_test, y_test)}')

def SVRRegr(X_train, y_train):
    """  Epsilon-Support Vector Regression
    """
    param_grid = {'C': [100, 10, 1],
                  'epsilon': [100, 10, 1, 0.1, 0.01, 0.001],
                  }
    grid = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\nНаилучшие параметры: {}".format(grid.best_params_))
    print("\nНаилучшие best_score_: {}".format(grid.best_score_))
    print(f'\nОценка результата обучения (SVR): {grid.score(X_train, y_train)}')
    accur = grid.score(X_test, y_test)
    print(f'Оценка результата предсказания (SVR): {accur}')

    save_model(grid, accur, 'svr')

def RidgeRegr(X_train, y_train, X_test, y_test):

    # print(X_train.shape)
    # select = SelectPercentile(percentile=70)
    # select.fit(X_train, y_train)
    # # преобразовываем обучающий набор
    # X_train_selected = select.transform(X_train)
    # print(X_train_selected.shape)
    # # преобразовываем тестовые данные
    # X_test_selected = select.transform(X_test)

    # pipe = Pipeline([('scaler', MinMaxScaler()), ("select", SelectPercentile()), ("ridge", Ridge())])
    pipe = Pipeline([("select", SelectPercentile()), ("ridge", Ridge())])
    # pipe = Pipeline([("ridge", Ridge())])
    param_grid = {
            'select__percentile': [20, 30, 40, 50, 60, 70, 80, 90, 95],
            'ridge__alpha': [10, 1, 0.1, 0.01, 0.001, 0.0001],
            'ridge__tol': [10, 1, 0.1, 0.01, 0.001, 0.0001],
            'ridge__random_state': [0]}
    grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\nНаилучшие параметры: {}".format(grid.best_params_))
    print("Наилучшие best_score_: {}".format(grid.best_score_))
    print(f'\nОценка результата обучения (Ridge): {grid.score(X_train, y_train)}')
    accur = grid.score(X_test, y_test)
    print(f'Оценка результата предсказания (Ridge): {accur}')

    save_model(grid, accur, 'ridge')


def RandomForestRegr(X_train, y_train):
    param_grid = {
                'max_depth': [4],
                #'max_depth': [2, 3, 4, 5],
                # 'max_features': [8],
                # 'max_features': [2,3,4,5,6,7,8,9,10],
                'max_features': [2,3,4,5,6,7,8,9,8],
                'random_state': [0]}
    grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\nНаилучшие параметры: {}".format(grid.best_params_))
    print(f'\nОценка результата обучения (RandomForestRegressor): {grid.score(X_train, y_train)}')
    accur = grid.score(X_test, y_test)
    print(f'Оценка результата предсказания (RandomForestRegressor): {accur}')

    save_model(grid, accur, 'forest_rnd')

# model_forest = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
# print(f'\nОценка результата обучения (GradientBoostingRegressor): {model_forest.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (GradientBoostingRegressor): {model_forest.score(X_test, y_test)}')
#
# from sklearn import tree
# dtree = tree.DecisionTreeRegressor(min_samples_split=20).fit(X_train, y_train)
# print(f'\nОценка результата обучения (DecisionTreeRegressor): {dtree.score(X_train, y_train)}')
# print(f'Оценка результата предсказания (DecisionTreeRegressor): {dtree.score(X_test, y_test)}')


def MLP_Regr(X_train, y_train):
    """ Multi-layer Perceptron regressor

    """
    # pipe = Pipeline([("scaler", MinMaxScaler()), ("mlp", MLPRegressor())])
    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor())])
    param_grid = {'mlp__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'mlp__max_iter': [1_000_000],
                # 'mlp__hidden_layer_sizes': [50, 100, 200],
                'mlp__activation': ('identity', 'logistic', 'tanh', 'relu'),
                'mlp__solver': ('sgd', 'adam'),
                'mlp__random_state': [0]}
    grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("\nНаилучшие параметры: {}".format(grid.best_params_))
    print(f'Оценка результата обучения (MLP), с нормализацией данных: {grid.score(X_train, y_train)}')
    accur =  grid.score(X_test, y_test)
    print(f'Оценка результата предсказания (MLP), с нормализацией данных: {accur}')
    save_model(grid, accur, 'mlpr')

def save_model(model, accur, pref=None):
    with open( f'{pref}_model_{accur}.pkl', 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    ds_diabetes = load_diabetes()
    X = ds_diabetes.data
    y = ds_diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForestRegr(X_train, y_train)
    # MLP_Regr(X_train, y_train)
    # RidgeRegr(X_train, y_train, X_test, y_test)
    # SVRRegr(X_train, y_train)

    # Исследование данных
    #
    # print(np.unique(X[:, 1]))
    # df_diabetes = pd.DataFrame(X, columns=ds_diabetes.feature_names)
    # print(df_diabetes.describe())
    # df_diabetes['target'] = ds_diabetes.target
    # df_diabetes['std_data'] = list(df_diabetes.std(axis=1))
    # print(df_diabetes)
    """
    axis=0 -- по X, axis=1 -- по Y
    """
    # print(df_diabetes.count(axis=0))

