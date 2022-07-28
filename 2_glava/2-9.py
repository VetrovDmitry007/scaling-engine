# Ансамбль деревьев решений на наборе данных "Breast Cancer". Визуализация важности признаков.
# 1. Случайный лес
# 2. Градиентный бустинг деревьев регрессии (Слабые ученики)
#
#  Оценки неопределённости для классификторов
#  1. Оценка вероятности прогноза -- predict_proba
#  1. Оценка определённости прогноза -- decision_function
#
# Интепретация модели при помощи библ. Shap
#
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import shap


def RandomForest():
    # X=(569, 30), y=(569,)
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    # max_features -- кол-во отбираемых признаков
    # n_estimators -- кол-во деревьев
    model_rand_forest = RandomForestClassifier(n_estimators=100, max_features=30, random_state=0).fit(X_train, y_train)
    print(f'Оценка результата обучения (RandomForestClassifier, n_estimators=100, max_features=30): {model_rand_forest.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (RandomForestClassifier, n_estimators=100, max_features=30): {model_rand_forest.score(X_test, y_test)}')

    model_rand_forest = RandomForestClassifier(n_estimators=100, max_features=10, max_depth=4, random_state=0).fit(X_train, y_train)
    print(f'\nОценка результата обучения (RandomForestClassifier, n_estimators=100, max_features=10, max_depth=4): {model_rand_forest.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (RandomForestClassifier, n_estimators=100, max_features=10, max_depth=4): {model_rand_forest.score(X_test, y_test)}')


    model_rand_forest = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    print(f'\nОценка результата обучения (RandomForestClassifier, n_estimators=100, max_features="auto"): {model_rand_forest.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (RandomForestClassifier, n_estimators=100, max_features="auto"): {model_rand_forest.score(X_test, y_test)}')

    # Визуализация важности признаков
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model_rand_forest.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    # plt.show()


def GradientBoosting():
    # X=(569, 30), y=(569,)
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    model_gb_0 = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    print(f'Оценка результата обучения (GradientBoostingClassifier): {model_gb_0.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (GradientBoostingClassifier): {model_gb_0.score(X_test, y_test)}')

    model_gb_1 = GradientBoostingClassifier(max_depth=1, random_state=0).fit(X_train, y_train)
    print(f'\nОценка результата обучения (GradientBoostingClassifier, max_depth=1): {model_gb_1.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (GradientBoostingClassifier, max_depth=1): {model_gb_1.score(X_test, y_test)}')

    model_gb_2 = GradientBoostingClassifier(learning_rate=0.01, random_state=0).fit(X_train, y_train)
    print(f'\nОценка результата обучения (GradientBoostingClassifier, learning_rate=0.01): {model_gb_2.score(X_train, y_train)}')
    print(f'Оценка результата предсказания (GradientBoostingClassifier, learning_rate=0.01): {model_gb_2.score(X_test, y_test)}')

    print(f'\nОценка вероятности прогноза:\n{model_gb_1.predict_proba(X_test[:6])}')
    # Положительное значение указывает на предпочтение в пользу позиционного класса (классу -- 1)
    print(f'\nОценка определённости прогноза:\n{model_gb_1.decision_function(X_test[:6])}')

    # return model_gb_1, X_test
    return model_gb_1, X_test, cancer

if __name__ == '__main__':
    # Случайный лес
    # RandomForest()
    # Градиентный бустинг деревьев регрессии ("Слабые ученики")
    best_model, X_test, cancer = GradientBoosting()

    # Интепретация модели при помощи библ. Shap
    df = pd.DataFrame(X_test, columns= cancer.feature_names)
    shap_values = shap.TreeExplainer(best_model).shap_values(df)
    # Интепритация конкретноо признака. Здесь -- 27-ой призн. из cancer.feature
    # shap.dependence_plot(27, shap_values, df)
    # Интепритация всех признаков
    shap.summary_plot(shap_values, df, max_display=10)


