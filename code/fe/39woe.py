from utils import *

num_agg = {"AdoptionSpeed": ["sum", "count"]}


def make_oof_woe(tr, te, num_folds, new_col):
    folds = GroupKFold(n_splits=num_folds)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(range(len(tr)), y=None, groups=rescuer_id)):
        tr_tr = tr.iloc[train_idx]
        tr_te = tr.iloc[valid_idx]

        agg = tr_tr.groupby(new_col).agg({**num_agg})
        agg.columns = pd.Index([e[0] + "_" + e[1].upper() + "_" + new_col for e in agg.columns.tolist()])
        sum_col = [c for c in agg.columns if "SUM" in c]
        count_col = [c.replace("SUM", "COUNT") for c in sum_col]

        event = agg[sum_col].values
        nonevent = agg[count_col].values - agg[sum_col].values
        dist_event = event / event.sum(axis=0)
        dist_nonevent = nonevent / nonevent.sum(axis=0)
        woe_col = [c.replace("SUM", "WOE") for c in sum_col]
        feats = pd.DataFrame(np.log(dist_event / dist_nonevent),
                             columns=woe_col)
        agg = pd.concat((agg.reset_index(), feats), axis=1)

        tr_te = pd.merge(tr_te, agg, how="left", on=new_col)
        if n_fold == 0:
            tt_likeli_ = tr_te[["PetID"] + woe_col]
        else:
            tt_likeli_ = pd.concat((tt_likeli_, tr_te[["PetID"] + woe_col]), axis=0)

    agg = tr.groupby(new_col).agg({**num_agg})
    agg.columns = pd.Index([e[0] + "_" + e[1].upper() + "_" + new_col for e in agg.columns.tolist()])
    sum_col = [c for c in agg.columns if "SUM" in c]
    count_col = [c.replace("SUM", "COUNT") for c in sum_col]

    event = agg[sum_col].values
    nonevent = agg[count_col].values - agg[sum_col].values
    dist_event = event / event.sum(axis=0)
    dist_nonevent = nonevent / nonevent.sum(axis=0)
    woe_col = [c.replace("SUM", "WOE") for c in sum_col]
    feats = pd.DataFrame(np.log(dist_event / dist_nonevent),
                         columns=woe_col)
    agg = pd.concat((agg.reset_index(), feats), axis=1)

    te = te.merge(agg.reset_index(), how="left", on=new_col)

    return tt_likeli_.reset_index(drop=True).append(te[["PetID"] + woe_col]).reset_index(drop=True)

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

test = train[len_train:]
train = train[:len_train]
rescuer_id = train["RescuerID"].values[:len_train]

orig_cols = train.columns
for c in categorical_features:
    woe = make_oof_woe(train, test, num_folds=5, new_col=c)
    train = pd.merge(train, woe, how="left", on="PetID")
new_cols = [c for c in train.columns if c not in orig_cols]

train[new_cols].to_feather("../feature/woe.feather")