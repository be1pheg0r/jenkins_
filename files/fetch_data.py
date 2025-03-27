from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def load_data():
    return fetch_california_housing()

def load_as_df():
    data = load_data()
    target = pd.DataFrame(data.target, columns=['target'])
    features = pd.DataFrame(data.data, columns=data.feature_names)
    return pd.concat([features, target], axis=1)

df = load_as_df()

path = os.path.join(os.getcwd(), 'data.csv')
df.to_csv(path, index=False)
