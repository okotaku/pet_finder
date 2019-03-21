from utils import *

def get_interactions(train):
    interaction_features = ['Age', 'Quantity', 'Fee', 'PhotoAmt', 'MaturitySize']
    for (c1, c2) in combinations(interaction_features, 2):
        train[c1 + '_mul_' + c2] = train[c1] * train[c2]
        train[c1 + '_div_' + c2] = train[c1] / train[c2]
    return train

orig_cols = train.columns
train = get_interactions(train)
new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("../feature/interactions.feather")