import pandas as pd
import feather

def load_datasets(feats):
    dfs = [pd.read_feather(f'./data/input/{f}.f') for f in feats]
    feats_mart = pd.concat(dfs, axis=1, sort=False)
    
    feats_train = feats_mart[:1913]
    feats_test  = feats_mart[1913:1941]
    
    return feats_train, feats_test


def load_target():
    y_train = pd.read_feather('./data/input/pre_sale_val.f')
    return y_train

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df