from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir('/var/jenkins_home/workspace/jenkins'))

def load_data():
    base_path = os.getcwd()
    file_path = os.path.join(base_path, 'data', 'data.csv')
    return pd.read_csv(file_path)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    y = df['target']
    quantiles = y.quantile([0.25, 0.5, 0.75])

    bins = [-float('inf'), quantiles[0.25], quantiles[0.5], quantiles[0.75], float('inf')]
    labels = [0, 1, 2, 3]

    df['target'] = pd.cut(y, bins=bins, labels=labels)
    return df

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

    print(f"Data saved to {file_path}")

save_data(df)
