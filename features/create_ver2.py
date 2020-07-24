import pandas as pd

from base_ver2 import Feature, get_arguments, generate_features

Feature.dir = 'features'

class family_size(Feature):
    def create_features(self):
        self.sales_train_validation = sales_train_validation.T


if __name__ == '__main__':
    args = get_arguments()

    sales_train_validation = pd.read_csv('data/input/sales_train_validation.csv')
#     test = pd.read_csv('input/test.csv')

    generate_features(globals(), args.force)