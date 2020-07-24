from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from nyaggle.feature_store import cached_feature
from tqdm import tqdm

@cached_feature("pre_sale_val")
def create_pre_sale_val(sales_train_validation):
    """trainとtestをくっつけたDataFrameを作る"""
    print("prepare pre_sale_val")
    pre_sale_val = sales_train_validation
    return pre_sale_val


if __name__ == "__main__":
    ROOT_DIR = ""
    INPUT_DIR = ROOT_DIR + "data/input/"
    sales_train_validation = pd.read_csv(INPUT_DIR + "sales_train_validation.csv")
#     fitting = pd.read_csv(INPUT_DIR / "fitting.csv")
#     submission = pd.read_csv(INPUT_DIR / "atmaCup5__sample_submission.csv")

    create_pre_sale_val(sales_train_validation)