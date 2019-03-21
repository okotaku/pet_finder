import GPy
import GPyOpt
import feather
import scipy as sp
import numpy as np
import pandas as pd
import lightgbm as lgb

from collections import Counter
from functools import partial
from math import sqrt
from scipy.stats import rankdata

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold, GroupKFold

import matplotlib.pyplot as plt
import seaborn as sns

target = 'AdoptionSpeed'
len_train = 14993
len_test = 3948

# ===============
# Params
# ===============
seed = 777
n_splits = 5
np.random.seed(seed)

# feature engineering
n_components = 5
img_size = 256
batch_size = 256

# model
MODEL_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'subsample': 0.9,
    'subsample_freq': 1,
    # 'colsample_bytree': 0.6,
    'max_depth': 9,
    'max_bin': 127,
    'reg_alpha': 0.11,
    'reg_lambda': 0.01,
    'min_child_weight': 0.2,
    'min_child_samples': 20,
    'min_gain_to_split': 0.02,
    'min_data_in_bin': 3,
    'bin_construct_sample_cnt': 5000,
    'cat_l2': 10,
    'verbose': -1,
    'nthread': 2,
    'seed': 777,
}
FIT_PARAMS = {
    'num_boost_round': 5000,
    'early_stopping_rounds': 100,
    'verbose_eval': 10000,
}

# define
maxvalue_dict = {}
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
    'Type',
    'Vaccinated',
    'Type_main_breed',
    'BreedName_main_breed',
    'Type_second_breed',
    'BreedName_second_breed',
]
numerical_features = []
text_features = ['Name', 'Description']
remove = ['index', 'seq_text', 'PetID', 'Name', 'Description', 'RescuerID', 'StateName', 'annots_top_desc',
          'sentiment_text',
          'Description_Emb', 'Description_bow', 'annots_top_desc_pick']


def get_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def get_y():
    return pd.read_csv('../input/train.csv', usecols=[target]).values.flatten()


def run_model(X_train, y_train, X_valid, y_valid, w_train, w_valid,
              categorical_features, numerical_features,
              predictors, maxvalue_dict, fold_id):
    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=predictors,
                        weight=w_train)
    valid = lgb.Dataset(X_valid, y_valid,
                        categorical_feature=categorical_features,
                        feature_name=predictors,
                        weight=w_valid)
    evals_result = {}
    model = lgb.train(
        MODEL_PARAMS,
        train,
        valid_sets=[valid],
        valid_names=['valid'],
        evals_result=evals_result,
        **FIT_PARAMS
    )

    # validation score
    y_pred_valid = model.predict(X_valid)

    # feature importances
    importances = pd.DataFrame()
    importances['feature'] = predictors
    importances['gain'] = model.feature_importance(importance_type='gain')
    importances['split'] = model.feature_importance(importance_type='split')
    importances['fold'] = fold_id

    return y_pred_valid, importances


def plot_mean_feature_importances(feature_importances, max_num=50, importance_type='gain', path=None):
    mean_gain = feature_importances[[importance_type, 'feature']].groupby('feature').mean()
    feature_importances['mean_' + importance_type] = feature_importances['feature'].map(mean_gain[importance_type])

    if path is not None:
        data = feature_importances.sort_values('mean_' + importance_type, ascending=False).iloc[:max_num, :]
        plt.clf()
        plt.figure(figsize=(16, 8))
        sns.barplot(x=importance_type, y='feature', data=data)
        plt.tight_layout()
        plt.savefig(path)

    return feature_importances


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


class OptimizedRounder_(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -get_score(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [1.5, 2.0, 2.5, 3.0]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(1, 2), (1.5, 2.5), (2, 3), (2.5, 3.5)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -get_score(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [0.2, 0.4, 0.6, 0.8]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(0.01, 0.3), (0.15, 0.56), (0.35, 0.75), (0.6, 0.9)]
        for it1 in range(10):
            for idx in range(4):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


class StratifiedGroupKFold():
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        fold = pd.DataFrame([X, y, groups]).T
        fold.columns = ['X', 'y', 'groups']
        fold['y'] = fold['y'].astype(int)
        g = fold.groupby('groups')['y'].agg('mean').reset_index()
        fold = fold.merge(g, how='left', on='groups', suffixes=('', '_mean'))
        fold['y_mean'] = fold['y_mean'].apply(np.round)
        fold['fold_id'] = 0
        for unique_y in fold['y_mean'].unique():
            mask = fold.y_mean == unique_y
            selected = fold[mask].reset_index(drop=True)
            cv = GroupKFold(n_splits=n_splits)
            for i, (train_index, valid_index) in enumerate(
                    cv.split(range(len(selected)), y=None, groups=selected['groups'])):
                selected.loc[valid_index, 'fold_id'] = i
            fold.loc[mask, 'fold_id'] = selected['fold_id'].values

        for i in range(self.n_splits):
            indices = np.arange(len(fold))
            train_index = indices[fold['fold_id'] != i]
            valid_index = indices[fold['fold_id'] == i]
            yield train_index, valid_index


def merge(train, test, path, add_cols):
    df_ = feather.read_dataframe(path)
    add_cols += list(df_.columns)
    train = pd.concat((train, df_[:len_train]), axis=1)
    test = pd.concat((test, df_[len_train:].reset_index(drop=True)), axis=1)
    return train, test, add_cols


train = feather.read_dataframe('from_kernel/all_data.feather')
test = train[len_train:]
train = train[:len_train]
add_cols = []

train, test, add_cols = merge(train, test, "feature/breedrank.feather", add_cols)

train, test, add_cols = merge(train, test, "feature/glove_wiki_mag_light.feather", add_cols)
train, test, add_cols = merge(train, test, "feature/glove_wiki_mag_light_meta.feather", add_cols)

# n_feats =2024
# predictors = list(data.feature[:n_feats])
use_cols = pd.read_csv("importance6.csv")
use_cols["gain"] = use_cols["gain"] / use_cols["gain"].sum()
predictors = list(use_cols[use_cols.gain > 0.0002].feature) + add_cols
print(len(predictors))
categorical_features = [c for c in categorical_features if c in predictors]
numerical_features = list(set(predictors) - set(categorical_features + [target] + remove))
# predictors = categorical_features + numerical_features

X = train.loc[:, predictors]
y = feather.read_dataframe('../input/X_train.feather')["AdoptionSpeed"].values
rescuer_id = pd.read_csv('../input/train.csv').loc[:, 'RescuerID'].iloc[:len_train]

labels2weight = {0: 1,
                 1: 1,
                 2: 1,
                 3: 1,
                 4: 1}


def training(x):
    print(x)
    feature_importances = pd.DataFrame()
    y_pred = np.empty(len_train, )
    y_test = []

    labels2weight = {0: float(x[:, 0]),
                     1: float(x[:, 1]),
                     2: float(x[:, 2]),
                     3: float(x[:, 3]),
                     4: float(x[:, 4])}
    weight = np.array([labels2weight[int(t)] for t in y])

    cv = StratifiedGroupKFold(n_splits=n_splits)
    for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)):
        X_train = X.loc[train_index, :]
        X_valid = X.loc[valid_index, :]
        y_train = y[train_index]
        y_valid = y[valid_index]
        w_train = weight[train_index]
        w_valid = weight[valid_index]

        y_pred_valid, importances = run_model(X_train, y_train, X_valid, y_valid, w_train, w_valid,
                                              categorical_features, numerical_features,
                                              predictors, maxvalue_dict, fold_id)
        y_pred_valid = rankdata(y_pred_valid) / len(y_pred_valid)
        y_pred[valid_index] = y_pred_valid.ravel()

    optR = OptimizedRounder()
    optR.fit(y_pred, y)
    coefficients = optR.coefficients()
    y_pred_opt = optR.predict(y_pred, coefficients)
    score = get_score(y, y_pred_opt)
    print(score)

    return -1 * score


bounds = [{'name': 'w0', 'type': 'continuous', 'domain': (0.1, 2)},
          {'name': 'w1', 'type': 'continuous', 'domain': (0.1, 2)},
          {'name': 'w2', 'type': 'continuous', 'domain': (0.1, 2)},
          {'name': 'w3', 'type': 'continuous', 'domain': (0.1, 2)},
          {'name': 'w4', 'type': 'continuous', 'domain': (0.1, 2)},
          ]
myBopt = GPyOpt.methods.BayesianOptimization(f=training, domain=bounds, initial_design_numdata=5, acquisition_type='EI')
myBopt.run_optimization(max_iter=150)

best_score = np.min(myBopt.Y)
best_itre = np.argmin(myBopt.Y)
result = pd.DataFrame({"param": myBopt.X[best_itre]},
                      index=["w0", "w1", "w2", "w3", "w4"])

print("best score", best_score)
print(result)