import feather
from category_encoders import *
from sklearn.model_selection import GroupKFold
from utils import *

categorical_features = [
     'Breed1',
     'Breed2',
     'Color1',
     'Color2',
     'Color3',
     'Dewormed',
     'FurLength',
     'Gender',
     'Health',
     'MaturitySize',
     'State',
     'Sterilized',
     'Vaccinated'
]

n_splits = 5
cv = GroupKFold(n_splits=n_splits)

for c in categorical_features:
    train[c] = train[c].astype("str") + "e"

test = train[len_train:]
train = train[:len_train]
y_train = train['AdoptionSpeed'].values[:len_train]
rescuer_id = train["RescuerID"].values[:len_train]
oof_enc = np.zeros((len(train), len(categorical_features)))
test_enc = np.zeros((len(test), len(categorical_features)))

for train_index, valid_index in cv.split(range(len(train)), y=None, groups=rescuer_id):
    X_tr, y_tr = train.iloc[train_index, :], y_train[train_index]
    X_val, y_val = train.iloc[valid_index, :], y_train[valid_index]

    enc = TargetEncoder(return_df=False, smoothing=1.0)
    enc.fit(X_tr[categorical_features].values, y_tr)

    oof_enc[valid_index] = enc.transform(X_val[categorical_features].values)
    test_enc += enc.transform(test[categorical_features].values) / n_splits

pd.DataFrame(oof_enc, columns=[c+"target_encode" for c in categorical_features]).to_feather("../feature/target_encode.feather")