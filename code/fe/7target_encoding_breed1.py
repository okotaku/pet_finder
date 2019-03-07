import feather
from category_encoders import *
from sklearn.model_selection import StratifiedKFold
from utils import *

train["Breed1"] = train["Breed1"].astype(str) + "e"
test["Breed1"] = test["Breed1"].astype(str) + "e"

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
y_train = train['AdoptionSpeed'].values
oof_enc = np.zeros(len(train))
test_enc = np.zeros(len(test))

for train_index, valid_index in kfold.split(train, y_train):
    X_tr, y_tr = train.iloc[train_index, :], y_train[train_index]
    X_val, y_val = train.iloc[valid_index, :], y_train[valid_index]

    enc = TargetEncoder(return_df=False, smoothing=1.0)
    enc.fit(X_tr["Breed1"].values, y_tr)

    oof_enc[valid_index] = enc.transform(X_val["Breed1"].values).reshape(-1)
    test_enc += enc.transform(test["Breed1"].values).reshape(-1) / n_splits

np.save("../feature/breed1_target_enc.npy", oof_enc)