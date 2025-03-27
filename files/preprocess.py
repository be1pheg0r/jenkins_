from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


def load_data():
    base_path = os.getcwd()
    file_path = os.path.join(base_path, 'data/data.csv')
    return pd.read_csv(file_path)


def encode_categorical(df):
    return pd.get_dummies(df, columns=['ocean_proximity'])


def encode_target(x: pd.Series) -> pd.Series:
    def get_quantile(x: pd.Series) -> int:
        quantiles = x.quantile([0.25, 0.5, 0.75])
        if x <= quantiles[0.25]:
            return 0
        elif x <= quantiles[0.5]:
            return 1
        elif x <= quantiles[0.75]:
            return 2
        else:
            return 3

    return x.apply(get_quantile)


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.apply(zscore) < 3).all(axis=1)]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    x = df.drop('target', axis=1)
    y = df['target']
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    return pd.concat([x, y], axis=1)


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop('target', axis=1)
    y = df['target']

    x = x.fillna(x.mean())
    y = y[x.index]

    df = pd.concat([x, y], axis=1)
    df.dropna(inplace=True)

    return df


def preprocess(df: pd.DataFrame, pipeline: tuple = (
        encode_categorical,
        encode_target,
        drop_outliers,
        normalize,
        fill_na
)) -> pd.DataFrame:
    for step in pipeline:
        df = step(df)
    return df

df = load_data()
df = preprocess(df)

def save_data(df):
    base_path = os.getcwd()
    file_path = os.path.join(base_path, 'data/preprocessed_data.csv')
    df.to_csv(file_path, index=False)

save_data(df)
