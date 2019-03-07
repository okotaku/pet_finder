from utils import *


with timer('aggregation'):
    train["log_Fee"] = np.log1p(train["Fee"])
    train["log_Age"] = np.log1p(train["Age"])
    orig_cols = train.columns
    stats = ['mean', 'sum', 'median', 'min', 'max']
    groupby_dict = [
        {
            'key': ['RescuerID'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['RescuerID', 'State'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['RescuerID', 'Type'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1', 'Breed2'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['State'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
        {
            'key': ['MaturitySize'],
            'var': ['log_Fee', 'log_Age'],
            'agg': stats
        },
    ]

    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train = groupby.transform(train)
    diff = DiffGroupbyTransformer(param_dict=groupby_dict)
    train = diff.transform(train)
    ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
    train = ratio.transform(train)
    new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("../feature/agg_log.feather")