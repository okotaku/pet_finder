import feather
from utils import *

var = ['tfidf_svd_{}'.format(i) for i in range(n_components)] \
        + ['tfidf_nmf_{}'.format(i) for i in range(n_components)] \
        + ['count_svd_{}'.format(i) for i in range(n_components)] \
        + ['count_nmf_{}'.format(i) for i in range(n_components)] \
        + ['tfidf2_svd_{}'.format(i) for i in range(n_components)] \
        + ['tfidf2_nmf_{}'.format(i) for i in range(n_components)] \
        + ['count2_svd_{}'.format(i) for i in range(n_components)] \
        + ['count2_nmf_{}'.format(i) for i in range(n_components)]
df = feather.read_dataframe("../X_train9.feather")[var]
df_ = feather.read_dataframe("../X_test9.feather")[var]
df = df.append(df_).reset_index(drop=True)
train = pd.concat([df, train.reset_index(drop=True)], axis=1)

with timer('aggregation'):
    orig_cols = train.columns
    stats = ['mean', 'sum', 'median', 'min', 'max']
    var = ['tfidf_svd_{}'.format(i) for i in range(n_components)] \
          + ['tfidf_nmf_{}'.format(i) for i in range(n_components)] \
          + ['count_svd_{}'.format(i) for i in range(n_components)] \
          + ['count_nmf_{}'.format(i) for i in range(n_components)] \
          + ['tfidf2_svd_{}'.format(i) for i in range(n_components)] \
          + ['tfidf2_nmf_{}'.format(i) for i in range(n_components)] \
          + ['count2_svd_{}'.format(i) for i in range(n_components)] \
          + ['count2_nmf_{}'.format(i) for i in range(n_components)]
    groupby_dict = [
        {
            'key': ['RescuerID'],
            'var': var,
            'agg': stats
        },
        {
            'key': ['RescuerID', 'Type'],
            'var': var,
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1', 'Breed2'],
            'var': var,
            'agg': stats
        },
        {
            'key': ['Type', 'Breed1'],
            'var': var,
            'agg': stats
        },
        {
            'key': ['State'],
            'var': var,
            'agg': stats
        },
        {
            'key': ['MaturitySize'],
            'var': var,
            'agg': stats
        },
    ]

    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train = groupby.transform(train)
    new_cols = [c for c in train.columns if c not in orig_cols]
    train[new_cols].to_feather("../feature/agg_tfidf.feather")