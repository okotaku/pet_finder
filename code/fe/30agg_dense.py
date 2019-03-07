import feather
from utils import *


df = feather.read_dataframe("../X_train9.feather")[['img_{}'.format(i) for i in range(256)]]
df_ = feather.read_dataframe("../X_test9.feather")[['img_{}'.format(i) for i in range(256)]]
df = df.append(df_).reset_index(drop=True)
train = pd.concat([df, train.reset_index(drop=True)], axis=1)


with timer('aggregation'):
    orig_cols = train.columns
    stats = ['mean', 'median', 'min', 'max', 'sum', 'var']
    groupby_dict = [
        {
            'key': ['RescuerID'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
        {
            'key': ['RescuerID', 'Type'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1', 'Breed2'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
        {
            'key': ['State'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
        {
            'key': ['MaturitySize'],
            'var': ['img_{}'.format(i) for i in range(256)],
            'agg': stats
        },
    ]

    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train = groupby.transform(train)
    new_cols = [c for c in train.columns if c not in orig_cols]
    train[new_cols].to_feather("../feature/agg_img.feather")