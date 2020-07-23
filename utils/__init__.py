import pandas as pd
import feather

def load_datasets(feats):
#     dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    dfs = [feather.read_dataframe(f'features/{f}_train.feather') for f in feats]

    X_train = pd.concat(dfs, axis=1, sort=False)
#     dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    dfs = [feather.read_dataframe(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df