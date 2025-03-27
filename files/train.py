from typing import Any

import sklearn
import pandas as pd
import os

import pickle


def load_data(path='../data/preprocessed_data.csv'):
    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, path)
    if not os.path.exists(path):
        e_p = '../data/data.csv'
        e_p = os.path.join(base_path, e_p)
        if not os.path.exists(e_p):
            raise FileNotFoundError(f'Could not find data at {path} or {e_p}')
        else:
            print(f'Could not find data at {path}, loading from {e_p}')

            df = pd.read_csv(e_p)
    df = pd.read_csv(path)

    return df


def split_data(df):
    y = df['target']
    X = df.drop(columns=['target'])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=52,
                                                                                shuffle=True)

    return X_train, X_test, y_train, y_test


def train_model(model: sklearn.model_selection, X_train, y_train, inplace=False) -> sklearn.model_selection:
    model.fit(X_train, y_train)
    if inplace:
        return None
    return model


def evaluate_model(model: sklearn.model_selection, eval_func: callable, X_test, y_test) -> Any:
    y_pred = model.predict(X_test)
    return eval_func(y_test, y_pred)


if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=52)
    model = train_model(model, X_train, y_train)


    eval_func = sklearn.metrics.classification_report
    report = evaluate_model(model, eval_func, X_test, y_test)
    print(report.split('\n'))

    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print('Model saved to models/model.pkl')


