import re
import nltk

nltk.download('punkt')
import feather
import pandas as pd
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPooling1D, concatenate, BatchNormalization
from keras.layers import Reshape, Flatten, Concatenate, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold
from pymagnitude import *
from keras.preprocessing.text import text_to_word_sequence


def analyzer(text):
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from',
                  'in', 'on']
    text = text.lower()  # 小文字化
    text = text.replace('\n', '')  # 改行削除
    text = text.replace('\t', '')  # タブ削除
    text = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', text)  # 記号をスペースに置き換え
    text = nltk.word_tokenize(text)

    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text = [stemmer.stem(t) for t in text]

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):  # 数字が含まれるものは除外
            continue
        if word in stop_words:  # ストップワードに含まれるものは除外
            continue
        if len(word) < 2:  # 1文字、0文字（空文字）は除外
            continue
        words.append(word)

    return " ".join(words)


def get_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


class OptimizedRounder(object):
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


def get_keras_data(df, description_embeds):
    X = {
        "numerical": df[numerical].values,
        "description": description_embeds,
        "img": df[img_cols]
    }
    for c in categorical_features:
        X[c] = df[c]
    return X


def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y - y_pred), axis=-1))


def get_model(max_features, embedding_dim, emb_n=5, dout=.4):
    inp_cats = []
    embs = []
    for c in categorical_features:
        inp_cat = Input(shape=[1], name=c)
        inp_cats.append(inp_cat)
        embs.append((Embedding(X_train[c].max() + 1, emb_n)(inp_cat)))
    cats = Flatten()(concatenate(embs))
    cats = Dense(8, activation="relu")(cats)
    cats = Dropout(dout)(cats)
    cats = BatchNormalization()(cats)

    inp_numerical = Input(shape=(len(numerical),), name="numerical")
    nums = Dense(128, activation="relu")(inp_numerical)
    nums = Dropout(dout)(nums)
    nums = BatchNormalization()(nums)

    inp_img = Input(shape=(len(img_cols),), name="img")
    x_img = BatchNormalization()(inp_img)

    inp_desc = Input(shape=(max_features, embedding_dim), name="description")
    emb_desc = SpatialDropout1D(0.3)(inp_desc)
    x1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(emb_desc)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    conc = BatchNormalization()(conc)

    x = concatenate([conc, x_img, nums, cats])
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dout / 2)(x)

    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp_cats + [inp_numerical, inp_img, inp_desc], outputs=out)
    model.compile(optimizer="adam", loss=rmse)
    return model


def w2v_pymagnitude_tonn(train_text, path, max_features):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    model = Magnitude(path)
    embedding_dim = model.dim

    result = []
    for text in train_corpus:
        vec = []
        for word in text:
            try:
                vec_ = model.query(word)
            except:
                continue
            vec.append(vec_)
        if len(vec) == 0:
            vec = np.zeros((max_features, embedding_dim))
        else:
            vec_ = [[0 for i in range(300)] for _ in range(max_features - len(vec))]
            vec_.extend(vec)
            vec = np.array(vec_)[:max_features]

        result.append(vec)

    return np.array(result), embedding_dim

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
max_features=128

X_train = feather.read_dataframe('X_train9.feather')
n_train = len(X_train)
img_cols = ["img_{}".format(i) for i in range(256)]
numerical = [c for c in X_train.columns if c not in categorical_features and c not in img_cols]

y =  feather.read_dataframe('../input/X_train.feather')["AdoptionSpeed"].values
rescuer_id = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').loc[:, 'RescuerID'].iloc[:n_train]

embedding = "../input/pymagnitude-data/glove.840B.300d.magnitude"
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
train[['Description', 'Name']] = train[['Description', 'Name']].astype(str)
train["Description"] = [analyzer(text) for text in train["Description"]]
X_desc, embedding_dim = w2v_pymagnitude_tonn(train["Description"][:n_train], embedding, max_features)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

for c in categorical_features:
    X_train[c] = LabelEncoder().fit_transform(X_train[c])
X_train.replace(np.inf, np.nan, inplace=True)
X_train.replace(-np.inf, np.nan, inplace=True)
X_train[numerical] = StandardScaler().fit_transform(X_train[numerical].rank())
X_train.fillna(0, inplace=True)

n_splits = 5
cv = GroupKFold(n_splits=n_splits)
avg_valid_kappa = 0
batch_size = 128
coeffs = None

# x_test = get_keras_data(test_df, desc_embs[len(train_df):])
# y_nn_test = np.zeros((len(test_df),))
y_nn_oof = np.zeros((X_train.shape[0]))

for train_idx, valid_idx in cv.split(range(len(X_train)), y=None, groups=rescuer_id):
    x_train = get_keras_data(X_train.iloc[train_idx], X_desc[train_idx])
    x_valid = get_keras_data(X_train.iloc[valid_idx], X_desc[valid_idx])
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = get_model(max_features, embedding_dim)
    # clr_tri = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=len(train_df)//batch_size, mode="triangular2")
    ckpt = ModelCheckpoint('model.hdf5', save_best_only=True,
                           monitor='val_loss', mode='min')
    history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid),
                        epochs=20, callbacks=[ckpt])
    model.load_weights('model.hdf5')

    y_nn_oof[valid_idx] = model.predict(x_valid, batch_size=1000).reshape(-1, )
    # y_nn_test += model.predict(x_test, batch_size=batch_size).reshape(-1,) / n_splits

optR = OptimizedRounder()
optR.fit(y_nn_oof, y)
coefficients = optR.coefficients()
y_nn_oof_opt = optR.predict(y_nn_oof, coefficients)
score = get_score(y, y_nn_oof_opt)
print(score)