from utils import *

orig_cols = train.columns
train = get_name_features(train)
new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].reset_index(drop=True).to_feather("../feature/name_features.feather")