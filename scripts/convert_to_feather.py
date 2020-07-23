import pandas as pd

target = [
    'calendar',
    'sales_train_evaluation',
    'sales_train_validation',
    'sample_submission',
    'sell_prices'
    
]

extension = 'csv'
# extension = 'tsv'
# extension = 'zip'

for t in target:
    (pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
