from utils import *


orig_cols = train.columns
stats = ['mean', 'sum', 'median', 'min', 'max']
groupby_dict = [
            {
                'key': ['RescuerID'],
                'var': ['Fee'],
                'agg': ['count'] + stats
            },
            {
                'key': ['RescuerID', 'State'],
                'var': ['Fee'],
                'agg': ['count'] + stats
            },
            {
                'key': ['RescuerID', 'Type'],
                'var': ['Fee'],
                'agg': ['count'] + stats
            },
            {
                'key': ['Type', 'Breed1', 'Breed2'],
                'var': ['Fee'],
                'agg': stats
            },
            {
                'key': ['Type', 'Breed1'],
                'var': ['Fee'],
                'agg': stats
            },
            {
                'key': ['State'],
                'var': ['Fee'],
                'agg': stats
            },
            {
                'key': ['MaturitySize'],
                'var': ['Fee'],
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

train[new_cols].to_feather("../feature/agg_v3.feather")