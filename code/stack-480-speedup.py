# -*- coding: utf-8 -*-
'''
- reordered process 
- replace num_images with PhotoAmt
'''
import itertools
import json
import gc
import glob
import multiprocessing
cpu_count = multiprocessing.cpu_count()
import os
import time
import cv2
import random
import re
import imagehash
import nltk
import lightgbm as lgb
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import tensorflow as tf
import torch
from scipy.stats import rankdata
from PIL import Image
from multiprocessing import Pool
from pymagnitude import Magnitude
from gensim.models.fasttext import FastText
from gensim.models import word2vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from contextlib import contextmanager
from itertools import combinations
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from keras import backend as K
from keras.applications.densenet import preprocess_input as preprocess_input_dense
from keras.applications.densenet import DenseNet121
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_incep
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPooling1D, concatenate, BatchNormalization, PReLU
from keras.layers import Reshape, Flatten, Concatenate, SpatialDropout1D, GlobalAveragePooling1D, Multiply
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ===============
# Constants
# ===============
COMPETITION_NAME = 'petfinder-adoption-prediction'
MODEL_NAME = 'v001'
logger = getLogger(COMPETITION_NAME)
LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'

target = 'AdoptionSpeed'
len_train = 14993
len_test = 3948

T_flag = True
K_flag = True
G_flag = True
debug = False
    
    
# ===============
# Params
# ===============
seed = 777
kaeru_seed = 1337
n_splits = 5
np.random.seed(seed)

# feature engineering
n_components = 5
n_components_gege_img = 32
n_components_gege_txt = 16
img_size = 256
batch_size = 256
batch_size_nn = 128

# feature engineering NN
max_len=128
n_important = 100

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
    'colsample_bytree': 0.6,
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
    'nthread': -1,
    'seed': 777,
}
KAERU_PARAMS = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 70,
          'max_depth': 9,
          'learning_rate': 0.01,
          'max_bin': 32,
          'bagging_freq': 2,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.02,
          'min_child_samples': 150,
          'min_child_weight': 0.02,
          'lambda_l2': 0.0475,
          'verbosity': -1,
          'seed': kaeru_seed}
ADV_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 64,
    'learning_rate': 0.02,
    'verbose': 0,
    'lambda_l1': 0.1,
    'seed': 1213
} 
MODEL_PARAMS_XGB = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'device': 'gpu',
    'silent': 1,
}
FIT_PARAMS = {
    'num_boost_round': 5000,
    'early_stopping_rounds': 100,
    'verbose_eval': 5000,
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
     'BreedName_main_breed_all',
]
contraction_mapping = {u"ain’t": u"is not", u"aren’t": u"are not",u"can’t": u"cannot", u"’cause": u"because",
                       u"could’ve": u"could have", u"couldn’t": u"could not", u"didn’t": u"did not",
                       u"doesn’t": u"does not", u"don’t": u"do not", u"hadn’t": u"had not",
                       u"hasn’t": u"has not", u"haven’t": u"have not", u"he’d": u"he would",
                       u"he’ll": u"he will", u"he’s": u"he is", u"how’d": u"how did", u"how’d’y": u"how do you",
                       u"how’ll": u"how will", u"how’s": u"how is",  u"I’d": u"I would",
                       u"I’d’ve": u"I would have", u"I’ll": u"I will", u"I’ll’ve": u"I will have",
                       u"I’m": u"I am", u"I’ve": u"I have", u"i’d": u"i would", u"i’d’ve": u"i would have",
                       u"i’ll": u"i will",  u"i’ll’ve": u"i will have",u"i’m": u"i am", u"i’ve": u"i have",
                       u"isn’t": u"is not", u"it’d": u"it would", u"it’d’ve": u"it would have",
                       u"it’ll": u"it will", u"it’ll’ve": u"it will have",u"it’s": u"it is",
                       u"let’s": u"let us", u"ma’am": u"madam", u"mayn’t": u"may not",
                       u"might’ve": u"might have",u"mightn’t": u"might not",u"mightn’t’ve": u"might not have",
                       u"must’ve": u"must have", u"mustn’t": u"must not", u"mustn’t’ve": u"must not have",
                       u"needn’t": u"need not", u"needn’t’ve": u"need not have",u"o’clock": u"of the clock",
                       u"oughtn’t": u"ought not", u"oughtn’t’ve": u"ought not have", u"shan’t": u"shall not", 
                       u"sha’n’t": u"shall not", u"shan’t’ve": u"shall not have", u"she’d": u"she would",
                       u"she’d’ve": u"she would have", u"she’ll": u"she will", u"she’ll’ve": u"she will have",
                       u"she’s": u"she is", u"should’ve": u"should have", u"shouldn’t": u"should not",
                       u"shouldn’t’ve": u"should not have", u"so’ve": u"so have",u"so’s": u"so as",
                       u"this’s": u"this is",u"that’d": u"that would", u"that’d’ve": u"that would have",
                       u"that’s": u"that is", u"there’d": u"there would", u"there’d’ve": u"there would have",
                       u"there’s": u"there is", u"here’s": u"here is",u"they’d": u"they would", 
                       u"they’d’ve": u"they would have", u"they’ll": u"they will", 
                       u"they’ll’ve": u"they will have", u"they’re": u"they are", u"they’ve": u"they have", 
                       u"to’ve": u"to have", u"wasn’t": u"was not", u"we’d": u"we would",
                       u"we’d’ve": u"we would have", u"we’ll": u"we will", u"we’ll’ve": u"we will have", 
                       u"we’re": u"we are", u"we’ve": u"we have", u"weren’t": u"were not",
                       u"what’ll": u"what will", u"what’ll’ve": u"what will have", u"what’re": u"what are",
                       u"what’s": u"what is", u"what’ve": u"what have", u"when’s": u"when is",
                       u"when’ve": u"when have", u"where’d": u"where did", u"where’s": u"where is",
                       u"where’ve": u"where have", u"who’ll": u"who will", u"who’ll’ve": u"who will have",
                       u"who’s": u"who is", u"who’ve": u"who have", u"why’s": u"why is", u"why’ve": u"why have",
                       u"will’ve": u"will have", u"won’t": u"will not", u"won’t’ve": u"will not have",
                       u"would’ve": u"would have", u"wouldn’t": u"would not", u"wouldn’t’ve": u"would not have",
                       u"y’all": u"you all", u"y’all’d": u"you all would",u"y’all’d’ve": u"you all would have",
                       u"y’all’re": u"you all are",u"y’all’ve": u"you all have",u"you’d": u"you would",
                       u"you’d’ve": u"you would have", u"you’ll": u"you will", u"you’ll’ve": u"you will have",
                       u"you’re": u"you are", u"you’ve": u"you have", u"cat’s": u"cat is",u" whatapp ": u" whatapps ",
                       u" whatssapp ": u" whatapps ",u" whatssap ": u" whatapps ",u" whatspp ": u" whatapps ",
                       u" whastapp ": u" whatapps ",u" whatsap ": u" whatapps ",u" whassap ": u" whatapps ",
                       u" watapps ": u" whatapps ",u"wetfood": u"wet food",u"intetested": u"interested",
                       u"领养条件，": u"领养条件",u"谢谢。": u"谢谢",
                       u"别打我，记住，我有反抗的牙齿，但我不会咬你。remember": u"别打我，记住，我有反抗的牙齿，但我不会咬你。",
                       u"有你。do": u"有你。",u"名字name": u"名字",u"year，": u"year",u"work，your": u"work your",
                       u"too，will": u"too will",u"timtams": u"timtam",u"spay。": u"spay",u"shoulder，a": u"shoulder a",
                       u"sherpherd": u"shepherd",u"sherphed": u"shepherd",u"sherperd": u"shepherd",
                       u"sherpard": u"shepherd",u"serious。": u"serious",u"remember，i": u"remember i",
                       u"recover，": u"recover",u"refundable指定期限内结扎后会全数奉还": u"refundable",
                       u"puchong区，有没有人有增添家庭成员？": u"puchong",u"puchong救的": u"puchong",
                       u"puchong，": u"puchong",u"month。": u"month",u"month，": u"month",
                       u"microchip（做狗牌一定要有主人的电话号码）": u"microchip",u"maju。": u"maju",u"maincoone": u"maincoon",
                       u"lumpur。": u"lumpur",u"location：阿里玛，大山脚": u"location",u"life🐾🐾": u"life",
                       u"kibble，": u"kibble",u"home…": u"home",u"hand，but": u"hand but",u"hair，a": u"hair a",
                       u"grey、brown": u"grey brown",u"gray，": u"gray",u"free免费": u"free",u"food，or": u"food or",
                       u"dog／dog": u"dog",u"dijumpa": u"dijumpai",u"dibela": u"dibelai",
                       u"beauuuuuuuuutiful": u"beautiful",u"adopt🙏": u"adopt",u"addopt": u"adopt",
                      u"enxiety": u"anxiety", u"vaksin": u"vaccine"}
numerical_features = []
text_features = ['Name', 'Description', 'Description_Emb', 'Description_bow']
meta_text =  ['BreedName_main_breed', 'BreedName_second_breed', 'annots_top_desc', 'sentiment_text', 
              'annots_top_desc_pick', 'sentiment_entities']
remove = ['index', 'seq_text', 'PetID', 'Name', 'Description', 'RescuerID', 'StateName', 'annots_top_desc','sentiment_text', 
          'sentiment_entities', 'Description_Emb', 'Description_bow', 'annots_top_desc_pick']
kaeru_drop_cols = ["2017GDPperCapita", "Bumiputra", "Chinese", "HDI", "Indian", "Latitude", "Longitude",
                   'color_red_score_mean_mean', 'color_red_score_mean_sum', 'color_blue_score_mean_mean',
                   'color_blue_score_mean_sum', 'color_green_score_mean_mean', 'color_green_score_mean_sum',
                   'dog_cat_scores_mean_mean', 'dog_cat_scores_mean_sum', 'dog_cat_topics_mean_mean',
                   'dog_cat_topics_mean_sum', 'is_dog_or_cat_mean_mean', 'is_dog_or_cat_mean_sum',
                   'len_text_mean_mean', 'len_text_mean_sum', 'StateID']
gege_drop_cols = ['2017GDPperCapita', 'Breed1_equals_Breed2', 'Bumiputra', 'Chinese',
                  'HDI', 'Indian','Latitude', 'Longitude', 'Pop_density', 'Urban_pop', 'Breed1_equals_Breed2',
                  'fix_Breed1', 'fix_Breed2', 'single_Breed', 'color_red_score_mean_mean', 'color_red_score_mean_sum', 
                  'color_red_score_mean_var',  'color_blue_score_mean_mean', 'color_blue_score_mean_sum', 
                   'color_blue_score_mean_var', 'color_green_score_mean_mean', 'color_green_score_mean_sum',
                  'color_green_score_mean_var', 'dog_cat_scores_mean_mean', 'dog_cat_scores_mean_sum', 
                  'dog_cat_scores_mean_var', 'dog_cat_topics_mean_mean', 'dog_cat_topics_mean_sum', 
                   'dog_cat_topics_mean_var', 'is_dog_or_cat_mean_mean', 'is_dog_or_cat_mean_sum',
                  'is_dog_or_cat_mean_var', 'len_text_mean_mean', 'len_text_mean_sum', 'len_text_mean_var']
use_cols = pd.read_csv("../input/pet-usecols/importance10.csv")
#use_cols = pd.read_csv("importance4.csv")
use_cols["gain"] = use_cols["gain"] / use_cols["gain"].sum()
use_colsnn = list(use_cols[use_cols.gain>0.0002].feature.values)
use_cols = list(use_cols.feature.values)[:1000]

ps = nltk.stem.PorterStemmer()
lc = nltk.stem.lancaster.LancasterStemmer()
sb = nltk.stem.snowball.SnowballStemmer('english')

# ===============
# Utility Functions
# ===============
def to_category(train, cat=None):
    if cat is None:
        cat = [col for col in train.columns if train[col].dtype == 'object']
    for c in cat:
        train[c], uniques = pd.factorize(train[c])
        maxvalue_dict[c] = train[c].max() + 1
    return train

def init_logger():
    # Add handlers
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter(LOGFORMAT))
    fh_handler = FileHandler('{}.log'.format(MODEL_NAME))
    fh_handler.setFormatter(Formatter(LOGFORMAT))
    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.addHandler(fh_handler)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_image_and_hash(paths):
    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
        #lambda x: imagehash.whash(x, mode='db4'),
    ]

    petids = []
    hashes = []
    for path in paths:
        image = Image.open(path)
        imageid = path.split('/')[-1].split('.')[0][:-2]

        petids.append(imageid)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))
    return petids, np.array(hashes).astype(np.uint8)
    
def find_duplicates_all():
    train_paths = glob.glob('../input/petfinder-adoption-prediction/train_images/*-1.jpg')
    train_paths += glob.glob('../input/petfinder-adoption-prediction/train_images/*-2.jpg')
    test_paths = glob.glob('../input/petfinder-adoption-prediction/test_images/*-1.jpg')
    test_paths += glob.glob('../input/petfinder-adoption-prediction/test_images/*-2.jpg')
    
    train_petids, train_hashes = load_image_and_hash(train_paths)
    test_petids, test_hashes = load_image_and_hash(test_paths)

#     sims = np.array([(train_hashes[i] == test_hashes).sum(axis=1)/256 for i in range(train_hashes.shape[0])])
    try:
        train_hashes = torch.from_numpy(train_hashes).cuda()
        test_hashes = torch.from_numpy(test_hashes).cuda()
    except:
        train_hashes = torch.Tensor(train_hashes).cuda()
        test_hashes = torch.Tensor(test_hashes).cuda()
    sims = np.array([(train_hashes[i] == test_hashes).sum(dim=1).cpu().numpy() for i in range(train_hashes.shape[0])]) / 256
    indices1 = np.where(sims > 0.9)
    indices2 = np.where(indices1[0] != indices1[1])
    petids1 = [train_petids[i] for i in indices1[0][indices2]]
    petids2 = [test_petids[i] for i in indices1[1][indices2]]
    dups = {tuple([petid1, petid2]): True for petid1, petid2 in zip(petids1, petids2)}
    logger.info(f'found {len(dups)} duplicates')

    return dups
    
def submission_with_postprocessV3(y_pred):
    df_sub = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
    train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    df_sub["AdoptionSpeed"] = y_pred
    # postprocess
    duplicated = find_duplicates_all()
    duplicated = pd.DataFrame(duplicated, index=range(0)).T.reset_index()
    duplicated.columns = ['pet_id_0', 'pet_id_1']
    duplicated.drop_duplicates(['pet_id_1'], inplace=True)

    duplicated_0 = duplicated.merge(train[['PetID', 'AdoptionSpeed']], how='left', left_on='pet_id_0', right_on='PetID').dropna()
    df_sub = df_sub.merge(duplicated_0[['pet_id_1', 'AdoptionSpeed']], 
                          how='left', left_on='PetID', right_on='pet_id_1', suffixes=('_original', ''))
    df_sub['AdoptionSpeed'].fillna(df_sub['AdoptionSpeed_original'], inplace=True)
    df_sub = df_sub[['PetID', 'AdoptionSpeed']]
    
    n = len(y_pred) - np.sum(y_pred==df_sub['AdoptionSpeed'])
    logger.info(f'changed {n} duplicates')
    
    df_sub["AdoptionSpeed"] = df_sub["AdoptionSpeed"].astype(np.int32)
    # submission
    df_sub.to_csv('submission.csv', index=False)
    
def submission(y_pred):
    logger.info('making submission file...')
    df_sub = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
    df_sub[target] = y_pred
    df_sub.to_csv('submission.csv', index=False)

def analyzer_bow(text):
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower() # 小文字化
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    for bad_word in contraction_mapping:
        if bad_word in text:
            text = text.replace(bad_word, contraction_mapping[bad_word])
    text = text.split(' ') # スペースで区切る
    text = [sb.stem(t) for t in text]
    
    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 2: #  1文字、0文字（空文字）は除外
            continue
        words.append(word)
        
    return " ".join(words)

def analyzer_embed(text):
    text = text.lower() # 小文字化
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    for bad_word in contraction_mapping:
        if bad_word in text:
            text = text.replace(bad_word, contraction_mapping[bad_word])
    text = text.split(' ') # スペースで区切る
    
    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if len(word) < 1: #  0文字（空文字）は除外
            continue
        words.append(word)
        
    return " ".join(words)

def analyzer_k(text):    
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower() # 小文字化
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    text = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', text) # 記号をスペースに置き換え
    text = text.split(' ') # スペースで区切る
    
    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 2: #  1文字、0文字（空文字）は除外
            continue
        words.append(word)
        
    return words
    
# ===============
# Feature Engineering
# ===============
class GroupbyTransformer():
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict['key']
        if 'var' in p_dict.keys():
            var = p_dict['var']
        else:
            var = self.var
        if 'agg' in p_dict.keys():
            agg = p_dict['agg']
        else:
            agg = self.agg
        if 'on' in p_dict.keys():
            on = p_dict['on']
        else:
            on = key
        return key, var, agg, on

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=on)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()

class DiffGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError
        
    def _merge(self):
        raise NotImplementedError
    
    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError
        
    def _merge(self):
        raise NotImplementedError
    
    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[v] / dataframe[base_feature]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['ratio', a, v, 'groupby'] + key) for v in var for a in _agg]
    
    
class CategoryVectorizer():
    def __init__(self, categorical_columns, n_components, 
                 vectorizer=CountVectorizer(), 
                 transformer=LatentDirichletAllocation(),
                 name='CountLDA'):
        self.categorical_columns = categorical_columns
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)
        
    def transform(self, dataframe):
        features = []
        for (col1, col2) in self.get_column_pairs():
            try:
                sentence = self.create_word_list(dataframe, col1, col2)
                sentence = self.vectorizer.fit_transform(sentence)
                feature = self.transformer.fit_transform(sentence)
                feature = self.get_feature(dataframe, col1, col2, feature, name=self.name)
                features.append(feature)
            except:
                pass
        features = pd.concat(features, axis=1)
        return features

    def create_word_list(self, dataframe, col1, col2):
        col1_size = int(dataframe[col1].values.max() + 1)
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(dataframe[col1].values, dataframe[col2].values):
            col2_list[int(val1)].append(col2+str(val2))
        return [' '.join(map(str, ls)) for ls in col2_list]
    
    def get_feature(self, dataframe, col1, col2, latent_vector, name=''):
        features = np.zeros(
            shape=(len(dataframe), self.n_components), dtype=np.float32)
        self.columns = ['_'.join([name, col1, col2, str(i)])
                   for i in range(self.n_components)]
        for i, val1 in enumerate(dataframe[col1]):
            features[i, :self.n_components] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=self.columns)
    
    def get_column_pairs(self):
        return [(col1, col2) for col1, col2 in itertools.product(self.categorical_columns, repeat=2) if col1 != col2]

    def get_numerical_features(self):
        return self.columns
    
class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X

# ===============
# For pet
# ===============
def merge_state_info(train):
    states = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
    state_info = pd.read_csv('../input/state-info/state_info.csv')
    state_info.rename(columns={
        'Area (km2)': 'Area', 
        'Pop. density': 'Pop_density',
        'Urban pop.(%)': 'Urban_pop',
        'Bumiputra (%)': 'Bumiputra',
        'Chinese (%)': 'Chinese',
        'Indian (%)': 'Indian'
    }, inplace=True)
    state_info['Population'] = state_info['Population'].str.replace(',', '').astype('int32')
    state_info['Area'] = state_info['Area'].str.replace(',', '').astype('int32')
    state_info['Pop_density'] = state_info['Pop_density'].str.replace(',', '').astype('int32')
    state_info['2017GDPperCapita'] = state_info['2017GDPperCapita'].str.replace(',', '').astype('float32')
    state_info['StateName'] = state_info['StateName'].str.replace('FT ', '')
    state_info['StateName'] = state_info['StateName'].str.replace('Malacca', 'Melaka')
    state_info['StateName'] = state_info['StateName'].str.replace('Penang', 'Pulau Pinang')

    states = states.merge(state_info, how='left', on='StateName')
    train = train.merge(states, how='left', left_on='State', right_on='StateID')
    
    return train    
    
def merge_breed_name(train):
    breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    with open("../input/petfinderdatasets/rating.json", 'r', encoding='utf-8') as f:
        breed_data = json.load(f)
    cat_breed = pd.DataFrame.from_dict(breed_data['cat_breeds']).T
    dog_breed = pd.DataFrame.from_dict(breed_data['dog_breeds']).T
    df = pd.concat([dog_breed, cat_breed], axis=0).reset_index().rename(columns={'index': 'BreedName'})
    df.BreedName.replace(
        {
            'Siamese Cat': 'Siamese',
            'Chinese Crested': 'Chinese Crested Dog',
            'Australian Cattle Dog': 'Australian Cattle Dog/Blue Heeler',
            'Yorkshire Terrier': 'Yorkshire Terrier Yorkie',
            'Pembroke Welsh Corgi': 'Welsh Corgi',
            'Sphynx': 'Sphynx (hairless cat)',
            'Plott': 'Plott Hound',
            'Korean Jindo Dog': 'Jindo',
            'Anatolian Shepherd Dog': 'Anatolian Shepherd',
            'Belgian Malinois': 'Belgian Shepherd Malinois',
            'Belgian Sheepdog': 'Belgian Shepherd Dog Sheepdog',
            'Belgian Tervuren': 'Belgian Shepherd Tervuren',
            'Bengal Cats': 'Bengal',
            'Bouvier des Flandres': 'Bouvier des Flanders',
            'Brittany': 'Brittany Spaniel',
            'Caucasian Shepherd Dog': 'Caucasian Sheepdog (Caucasian Ovtcharka)',
            'Dandie Dinmont Terrier': 'Dandi Dinmont Terrier',
            'Bulldog': 'English Bulldog',
            'American English Coonhound': 'English Coonhound',
            'Small Munsterlander Pointer': 'Munsterlander',
            'Entlebucher Mountain Dog': 'Entlebucher',
            'Exotic': 'Exotic Shorthair',
            'Flat-Coated Retriever': 'Flat-coated Retriever',
            'English Foxhound': 'Foxhound',
            'Alaskan Klee Kai': 'Klee Kai',
            'Newfoundland': 'Newfoundland Dog',
            'Norwegian Forest': 'Norwegian Forest Cat',
            'Nova Scotia Duck Tolling Retriever': 'Nova Scotia Duck-Tolling Retriever',
            'American Pit Bull Terrier': 'Pit Bull Terrier',
            'Ragdoll Cats': 'Ragdoll',
            'Standard Schnauzer': 'Schnauzer',
            'Scottish Terrier': 'Scottish Terrier Scottie',
            'Chinese Shar-Pei': 'Shar Pei',
            'Shetland Sheepdog': 'Shetland Sheepdog Sheltie',
            'West Highland White Terrier': 'West Highland White Terrier Westie',
            'Soft Coated Wheaten Terrier': 'Wheaten Terrier',
            'Wirehaired Pointing Griffon': 'Wire-haired Pointing Griffon',
            'Xoloitzcuintli': 'Wirehaired Terrier',
            'Cane Corso': 'Cane Corso Mastiff',
            'Havana Brown': 'Havana',
        }, inplace=True
        )
    breeds = breeds.merge(df, how='left', on='BreedName')
    
    breeds1_dic, breeds2_dic = {}, {}
    for c in breeds.columns:
        if c == "BreedID":
            continue
        breeds1_dic[c] = c + "_main_breed_all"
        breeds2_dic[c] = c + "_second_breed_all"
    train = train.merge(breeds.rename(columns=breeds1_dic), how='left', left_on='Breed1', right_on='BreedID')
    train.drop(['BreedID'], axis=1, inplace=True)
    train = train.merge(breeds.rename(columns=breeds2_dic), how='left', left_on='Breed2', right_on='BreedID')
    train.drop(['BreedID'], axis=1, inplace=True)
    
    return train

def merge_breed_name_sub(train):
    breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    df = pd.read_json('../input/petfinderdatasets/rating.json')
    cat_df = df.cat_breeds.dropna(0).reset_index().rename(columns={'index': 'BreedName'})
    dog_df = df.dog_breeds.dropna(0).reset_index().rename(columns={'index': 'BreedName'})

    cat = cat_df['cat_breeds'].apply(lambda x: pd.Series(x))
    cat_df = pd.concat([cat_df, cat], axis=1).drop(['cat_breeds'], axis=1)
    dog = dog_df['dog_breeds'].apply(lambda x: pd.Series(x))
    dog_df = pd.concat([dog_df, cat], axis=1).drop(['dog_breeds'], axis=1)

    df = pd.concat([dog_df, cat_df])
    df.BreedName.replace(
        {
            'Siamese Cat': 'Siamese',
            'Chinese Crested': 'Chinese Crested Dog',
            'Australian Cattle Dog': 'Australian Cattle Dog/Blue Heeler',
            'Yorkshire Terrier': 'Yorkshire Terrier Yorkie',
            'Pembroke Welsh Corgi': 'Welsh Corgi',
            'Sphynx': 'Sphynx (hairless cat)',
            'Plott': 'Plott Hound',
            'Korean Jindo Dog': 'Jindo',
            'Anatolian Shepherd Dog': 'Anatolian Shepherd',
            'Belgian Malinois': 'Belgian Shepherd Malinois',
            'Belgian Sheepdog': 'Belgian Shepherd Dog Sheepdog',
            'Belgian Tervuren': 'Belgian Shepherd Tervuren',
            'Bengal Cats': 'Bengal',
            'Bouvier des Flandres': 'Bouvier des Flanders',
            'Brittany': 'Brittany Spaniel',
            'Caucasian Shepherd Dog': 'Caucasian Sheepdog (Caucasian Ovtcharka)',
            'Dandie Dinmont Terrier': 'Dandi Dinmont Terrier',
            'Bulldog': 'English Bulldog',
            'American English Coonhound': 'English Coonhound',
            'Small Munsterlander Pointer': 'Munsterlander',
            'Entlebucher Mountain Dog': 'Entlebucher',
            'Exotic': 'Exotic Shorthair',
            'Flat-Coated Retriever': 'Flat-coated Retriever',
            'English Foxhound': 'Foxhound',
            'Alaskan Klee Kai': 'Klee Kai',
            'Newfoundland': 'Newfoundland Dog',
            'Norwegian Forest': 'Norwegian Forest Cat',
            'Nova Scotia Duck Tolling Retriever': 'Nova Scotia Duck-Tolling Retriever',
            'American Pit Bull Terrier': 'Pit Bull Terrier',
            'Ragdoll Cats': 'Ragdoll',
            'Standard Schnauzer': 'Schnauzer',
            'Scottish Terrier': 'Scottish Terrier Scottie',
            'Chinese Shar-Pei': 'Shar Pei',
            'Shetland Sheepdog': 'Shetland Sheepdog Sheltie',
            'West Highland White Terrier': 'West Highland White Terrier Westie',
            'Soft Coated Wheaten Terrier': 'Wheaten Terrier',
            'Wirehaired Pointing Griffon': 'Wire-haired Pointing Griffon',
            'Xoloitzcuintli': 'Wirehaired Terrier',
            'Cane Corso': 'Cane Corso Mastiff',
            'Havana Brown': 'Havana',
        }, inplace=True
        )
    breeds = breeds.merge(df, how='left', on='BreedName')

    train = train.merge(breeds.rename(columns={'BreedName': 'BreedName_main_breed'}), how='left', left_on='Breed1', right_on='BreedID', suffixes=('', '_main_breed'))
    train.drop(['BreedID'], axis=1, inplace=True)
    train = train.merge(breeds.rename(columns={'BreedName': 'BreedName_second_breed'}), how='left', left_on='Breed2', right_on='BreedID', suffixes=('', '_second_breed'))
    train.drop(['BreedID'], axis=1, inplace=True)
    
    return train
      
def extract_emojis(text, emoji_list):
    return ' '.join(c for c in text if c in emoji_list)
    
def merge_emoji(train):
    emoji = pd.read_csv('../input/petfinderdatasets/Emoji_Sentiment_Data_v1.0.csv')
    emoji2 = pd.read_csv('../input/petfinderdatasets/Emojitracker_20150604.csv')
    emoji = emoji.merge(emoji2, how='left', on='Emoji', suffixes=('', '_tracker'))
    
    emoji_list = emoji['Emoji'].values
    train_emoji = train['Description'].apply(extract_emojis, emoji_list=emoji_list)
    train_emoji = pd.DataFrame([train['PetID'], train_emoji]).T.set_index('PetID')
    train_emoji = train_emoji['Description'].str.extractall('(' + ')|('.join(emoji_list) + ')')
    train_emoji = train_emoji.fillna(method='bfill', axis=1).iloc[:, 0].reset_index().rename(columns={0: 'Emoji'})
    train_emoji = train_emoji.merge(emoji, how='left', on='Emoji')
    
    emoji_columns = ['Occurrences', 'Position', 'Negative', 'Neutral', 'Positive', 'Occurrences_tracker']
    stats = ['mean', 'max', 'min', 'median', 'std']
    g = train_emoji.groupby('PetID')[emoji_columns].agg(stats)
    g.columns = [c + '_' + stat for c in emoji_columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')
    
    return train

def get_interactions(train):
    interaction_features = ['Age', 'Quantity']
    for (c1, c2) in combinations(interaction_features, 2):
        train[c1 + '_mul_' + c2] = train[c1] * train[c2]
        train[c1 + '_div_' + c2] = train[c1] / train[c2]
    return train

def get_text_features(train):
    train['Length_Description'] = train['Description'].map(len)
    train['Length_annots_top_desc'] = train['annots_top_desc'].map(len)
    train['Lengths_sentiment_text'] = train['sentiment_text'].map(len)
    train['Lengths_sentiment_entities'] = train['sentiment_entities'].map(len)
    
    return train

def get_name_features(train):
    train['num_name_chars'] = train['Name'].apply(len)
    train['num_name_capitals'] = train['Name'].apply(lambda x: sum(1 for c in x if c.isupper()))
    train['name_caps_vs_length'] = train.apply(lambda row: row['num_name_capitals'] / (row['num_name_chars']+1e-5), axis=1)
    train['num_name_exclamation_marks'] = train['Name'].apply(lambda x: x.count('!'))
    train['num_name_question_marks'] = train['Name'].apply(lambda x: x.count('?'))
    train['num_name_punctuation'] = train['Name'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    train['num_name_symbols'] = train['Name'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    train['num_name_words'] = train['Name'].apply(lambda x: len(x.split()))
    return train

class MetaDataParser(object):   
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        # sentiment files
        train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))
        test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))
        sentiment_files = train_sentiment_files + test_sentiment_files
        self.sentiment_files = pd.DataFrame(sentiment_files, columns=['sentiment_filename'])
        self.sentiment_files['PetID'] = self.sentiment_files['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
        
        # metadata files
        train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
        test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))
        metadata_files = train_metadata_files + test_metadata_files
        self.metadata_files = pd.DataFrame(metadata_files, columns=['metadata_filename'])
        self.metadata_files['PetID'] = self.metadata_files['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    
    def open_json_file(self, filename):
        with open(filename, 'r', encoding="utf-8") as f:
            metadata_file = json.load(f)
        return metadata_file
    
    def get_stats(self, array, name):
        stats = [np.mean, np.max, np.min, np.sum, np.var]
        result = {}
        if len(array):
            for stat in stats:
                result[name + '_' + stat.__name__] = stat(array)
        else:
            for stat in stats:
                result[name + '_' + stat.__name__] = 0
        return result
            
    def parse_sentiment_file(self, file):
        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = ' '.join(file_entities)

        file_sentences_text = [x['text']['content'] for x in file['sentences']]
        file_sentences_text = ' '.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment_sum = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment_sum = file_sentences_sentiment_sum.add_prefix('document_sum_').to_dict()
        
        file_sentences_sentiment_mean = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').mean()
        file_sentences_sentiment_mean = file_sentences_sentiment_mean.add_prefix('document_mean_').to_dict()
        
        file_sentences_sentiment_var = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment_var = file_sentences_sentiment_var.add_prefix('document_var_').to_dict()
        
        file_sentiment.update(file_sentences_sentiment_mean)
        file_sentiment.update(file_sentences_sentiment_sum)
        file_sentiment.update(file_sentences_sentiment_var)
        file_sentiment.update({"sentiment_text": file_sentences_text})
        file_sentiment.update({"sentiment_entities": file_entities})
        
        return pd.Series(file_sentiment)
    
    def parse_metadata(self, file):
        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            label_annotations = file['labelAnnotations']
            file_top_score = [x['score'] for x in label_annotations]
            pick_value = int(len(label_annotations) * 0.3)
            if pick_value == 0: pick_value = 1 
            file_top_score_pick = [x['score'] for x in label_annotations[:pick_value]]
            file_top_desc = [x['description'] for x in label_annotations]       
            file_top_desc_pick = [x['description'] for x in label_annotations[:pick_value]]   
            dog_cat_scores = []
            dog_cat_topics = []
            is_dog_or_cat = []
            for label in label_annotations:
                if label['description'] == 'dog' or label['description'] == 'cat':
                    dog_cat_scores.append(label['score'])
                    dog_cat_topics.append(label['topicality'])
                    is_dog_or_cat.append(1)  
                else:
                    is_dog_or_cat.append(0) 
        else:
            file_top_score = []
            file_top_desc = []
            dog_cat_scores = []
            dog_cat_topics = []
            is_dog_or_cat = []
            file_top_score_pick = []
            file_top_desc_pick = []
            
        if 'faceAnnotations' in file_keys:
            file_face = file['faceAnnotations']
            n_faces = len(file_face)
        else:
            n_faces = 0
        
        if 'textAnnotations' in file_keys:
            text_annotations = file['textAnnotations']
            file_n_text_annotations = len(text_annotations)
            file_len_text = [len(text['description']) for text in text_annotations]                
        else:
            file_n_text_annotations = 0
            file_len_text = []

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = [x['score'] for x in file_colors]
        file_color_pixelfrac = [x['pixelFraction'] for x in file_colors]
        file_color_red = [x['color']['red'] if 'red' in x['color'].keys() else 0 for x in file_colors]
        file_color_blue = [x['color']['blue'] if 'blue' in x['color'].keys() else 0 for x in file_colors]
        file_color_green = [x['color']['green'] if 'green' in x['color'].keys() else 0 for x in file_colors]
        file_crop_conf = np.mean([x['confidence'] for x in file_crops])
        file_crop_x = np.mean([x['boundingPoly']['vertices'][1]['x'] for x in file_crops])
        file_crop_y = np.mean([x['boundingPoly']['vertices'][3]['y'] for x in file_crops])

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.mean([x['importanceFraction'] for x in file_crops])
        else:
            file_crop_importance = 0

        metadata = {
            'annots_top_desc': ' '.join(file_top_desc),
            'annots_top_desc_pick': ' '.join(file_top_desc_pick),
            'annots_score_pick_mean': np.mean(file_top_score_pick),
            'n_faces': n_faces,
            'n_text_annotations': file_n_text_annotations,
            'crop_conf': file_crop_conf,
            'crop_x': file_crop_x,
            'crop_y': file_crop_y,
            'crop_importance': file_crop_importance,
        }
        metadata.update(self.get_stats(file_top_score, 'annots_score_normal'))
        metadata.update(self.get_stats(file_color_score, 'color_score'))
        metadata.update(self.get_stats(file_color_pixelfrac, 'color_pixel_score'))
        metadata.update(self.get_stats(file_color_red, 'color_red_score'))
        metadata.update(self.get_stats(file_color_blue, 'color_blue_score'))
        metadata.update(self.get_stats(file_color_green, 'color_green_score'))
        metadata.update(self.get_stats(dog_cat_scores, 'dog_cat_scores'))
        metadata.update(self.get_stats(dog_cat_topics, 'dog_cat_topics'))
        metadata.update(self.get_stats(is_dog_or_cat, 'is_dog_or_cat'))
        metadata.update(self.get_stats(file_len_text, 'len_text'))
        metadata.update({"color_red_score_first": file_color_red[0] if len(file_color_red)>0 else -1})
        metadata.update({"color_blue_score_first": file_color_blue[0] if len(file_color_blue)>0 else -1})
        metadata.update({"color_green_score_first": file_color_green[0] if len(file_color_green)>0 else -1})
        metadata.update({"color_pixel_score_first": file_color_pixelfrac[0] if len(file_color_pixelfrac)>0 else -1})
        metadata.update({"color_score_first": file_color_score[0] if len(file_color_score)>0 else -1})
        metadata.update({"label_score_first": file_top_score[0] if len(file_top_score)>0 else -1})

        return pd.Series(metadata)
    
    def _transform(self, path, sentiment=True):
        file = self.open_json_file(path)
        if sentiment:
            result = self.parse_sentiment_file(file)
        else:
            result = self.parse_metadata(file)
        return result
    
    def process_sentiment(self, df):
        return df.apply(lambda x: self._transform(x, sentiment=True))
    
    def process_metadata(self, df):
        return df.apply(lambda x: self._transform(x, sentiment=False))
        
    def transform_sentiment(self):
        with multiprocessing.Pool(processes=self.n_jobs) as p:
            split_dfs = np.array_split(self.sentiment_files['sentiment_filename'], self.n_jobs)
            pool_results = p.map(self.process_sentiment, split_dfs)
        self.sentiment_features = pd.concat(pool_results)
        self.sentiment_files = pd.concat([self.sentiment_files, self.sentiment_features], axis=1, sort=False)
        
    def transform_metadata(self):
        with multiprocessing.Pool(processes=self.n_jobs) as p:
            split_dfs = np.array_split(self.metadata_files['metadata_filename'], self.n_jobs)
            pool_results = p.map(self.process_metadata, split_dfs)
        self.metadata_features = pd.concat(pool_results)
        self.metadata_files = pd.concat([self.metadata_files, self.metadata_features], axis=1, sort=False)
        
    def transform(self, train):
        self.transform_sentiment()
        stats = ['mean']
        columns = [c for c in self.sentiment_features.columns if c not in ['sentiment_text', 'sentiment_entities']]
        self.g1 = self.sentiment_files[list(self.sentiment_features.columns) + ['PetID']].groupby('PetID').agg(stats)
        self.g1.columns = [c + '_' + stat for c in columns for stat in stats]
        train = train.merge(self.g1, how='left', on='PetID')

        self.transform_metadata()
        stats = ['mean', 'min', 'max', 'median', 'var', 'sum', 'first']
        columns = [c for c in self.metadata_features.columns if c not in ['annots_top_desc', 'annots_top_desc_pick']]
        self.g2 = self.metadata_files[columns + ['PetID']].groupby('PetID').agg(stats)
        self.g2.columns = [c + '_' + stat for c in columns for stat in stats]
        train = train.merge(self.g2, how='left', on='PetID')
        
        return train

def pretrained_w2v(train_text, model, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    result = []
    for text in train_corpus:
        n_skip = 0
        vec = np.zeros(model.vector_size)
        for n_w, word in enumerate(text):
            if word in model: #0.9906
                vec = vec + model.wv[word]
                continue
            word_ = word.upper()
            if word_ in model: #0.9909
                vec = vec + model.wv[word_]
                continue
            word_ = word.capitalize()
            if word_ in model: #0.9925
                vec = vec + model.wv[word_]
                continue
            word_ = ps.stem(word)
            if word_ in model: #0.9927
                vec = vec + model.wv[word_]
                continue
            word_ = lc.stem(word)
            if word_ in model: #0.9932
                vec = vec + model.wv[word_]
                continue
            word_ = sb.stem(word)
            if word_ in model: #0.9933
                vec = vec + model.wv[word_]
                continue
            else:
                n_skip += 1
                continue
        vec = vec / (n_w - n_skip + 1)
        result.append(vec)

    w2v_cols = ["{}{}".format(name, i) for i in range(1, model.vector_size+1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols
    
    return result

def w2v_pymagnitude(train_text, model, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    
    result = []
    for text in train_corpus:
        vec = np.zeros(model.dim)
        for n_w, word in enumerate(text):
            if word in model: #0.9906
                vec = vec + model.query(word)
                continue
            word_ = word.upper()
            if word_ in model: #0.9909
                vec = vec + model.query(word_)
                continue
            word_ = word.capitalize()
            if word_ in model: #0.9925
                vec = vec + model.query(word_)
                continue
            word_ = ps.stem(word)
            if word_ in model: #0.9927
                vec = vec + model.query(word_)
                continue
            word_ = lc.stem(word)
            if word_ in model: #0.9932
                vec = vec + model.query(word_)
                continue
            word_ = sb.stem(word)
            if word_ in model: #0.9933
                vec = vec + model.query(word_)
                continue
            vec = vec + model.query(word)
                
        vec = vec / (n_w + 1)
        result.append(vec)

    w2v_cols = ["{}_mag{}".format(name, i) for i in range(1, model.dim+1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols
    
    return result

def doc2vec(description_k, d2v_param):
    corpus = [TaggedDocument(words=analyzer_k(text), tags=[i]) for i, text in enumerate(description_k)]
    doc2vecs = Doc2Vec(
        documents=corpus, dm=1, 
        **d2v_param
    ) # dm == 1 -> dmpv, dm != 1 -> DBoW
    doc2vecs = np.array([doc2vecs.infer_vector(analyzer_k(text)) for text in description_k])
    
    doc2vec_df = pd.DataFrame()
    doc2vec_df['d2v_mean'] = np.mean(doc2vecs, axis=1)
    doc2vec_df['d2v_sum'] = np.sum(doc2vecs, axis=1)
    doc2vec_df['d2v_max'] = np.max(doc2vecs, axis=1)
    doc2vec_df['d2v_min'] = np.min(doc2vecs, axis=1)
    doc2vec_df['d2v_median'] = np.median(doc2vecs, axis=1)
    doc2vec_df['d2v_var'] = np.var(doc2vecs, axis=1)
    
    return doc2vec_df

def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path):
    image = cv2.imread(path)
    new_image = resize_to_square(image)
    return preprocess_input_dense(new_image), preprocess_input_incep(new_image)

def get_age_feats(df):
    df["Age_year"] = (df["Age"] / 12).astype(np.int32)
    over_1year_flag = df["Age"] / 12 >= 1
    df.loc[over_1year_flag, "over_1year"] = 1
    df.loc[~over_1year_flag, "over_1year"] = 0
    return df

def freq_encoding(df, freq_cols):
    for c in freq_cols:
        count_df = df.groupby([c])['PetID'].count().reset_index()
        count_df.columns = [c, '{}_freq'.format(c)]
        df = df.merge(count_df, how='left', on=c)
        
    return df

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size 
    
# ===============
# Model
# ===============
def get_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def get_y():
    return pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv', usecols=[target]).values.flatten()
    
def run_model(X_train, y_train, X_valid, y_valid, X_test,
            categorical_features,
            predictors, maxvalue_dict, fold_id, params, model_name):
    train = lgb.Dataset(X_train, y_train, 
                        categorical_feature=categorical_features, 
                        feature_name=predictors)
    valid = lgb.Dataset(X_valid, y_valid, 
                        categorical_feature=categorical_features, 
                        feature_name=predictors)
    evals_result = {}
    model = lgb.train(
        params,
        train,
        valid_sets=[valid],
        valid_names=['valid'],
        evals_result=evals_result,
        **FIT_PARAMS
    )
    logger.info(f'Best Iteration: {model.best_iteration}')
    
    # train score
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # validation score
    y_pred_valid = model.predict(X_valid)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    y_pred_valid = rankdata(y_pred_valid)/len(y_pred_valid)

    # save model
    model.save_model(f'{model_name}_fold{fold_id}.txt')

    # predict test
    y_pred_test = model.predict(X_test)
    y_pred_test = rankdata(y_pred_test)/len(y_pred_test)

    # save predictions
    np.save(f'{model_name}_train_fold{fold_id}.npy', y_pred_valid)
    np.save(f'{model_name}_test_fold{fold_id}.npy', y_pred_test)

    return y_pred_valid, y_pred_test, train_rmse, valid_rmse

def run_xgb_model(X_train, y_train, X_valid, y_valid, X_test,
            predictors, maxvalue_dict, fold_id, params, model_name):
    d_train = xgb.DMatrix(data=X_train, label=y_train, feature_names=predictors)
    d_valid = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=predictors)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(dtrain=d_train, evals=watchlist, params=params, **FIT_PARAMS)
    
    # train score
    y_pred_train = model.predict(d_train, ntree_limit=model.best_ntree_limit)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # validation score
    y_pred_valid = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    y_pred_valid = rankdata(y_pred_valid)/len(y_pred_valid)

    # save model
    model.save_model(f'{model_name}_fold{fold_id}.txt')

    # predict test
    y_pred_test = model.predict(xgb.DMatrix(data=X_test, feature_names=predictors), ntree_limit=model.best_ntree_limit)
    y_pred_test = rankdata(y_pred_test)/len(y_pred_test)

    # save predictions
    np.save(f'{model_name}_train_fold{fold_id}.npy', y_pred_valid)
    np.save(f'{model_name}_test_fold{fold_id}.npy', y_pred_test)

    return y_pred_valid, y_pred_test, train_rmse, valid_rmse
 
def plot_mean_feature_importances(feature_importances, max_num=50, importance_type='gain', path=None):
    mean_gain = feature_importances[[importance_type, 'feature']].groupby('feature').mean()
    feature_importances['mean_' + importance_type] = feature_importances['feature'].map(mean_gain[importance_type])

    if path is not None:
        data = feature_importances.sort_values('mean_'+importance_type, ascending=False).iloc[:max_num, :]
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
            mask = fold.y_mean==unique_y
            selected = fold[mask].reset_index(drop=True)
            cv = GroupKFold(n_splits=n_splits)
            for i, (train_index, valid_index) in enumerate(cv.split(range(len(selected)), y=None, groups=selected['groups'])):
                selected.loc[valid_index, 'fold_id'] = i
            fold.loc[mask, 'fold_id'] = selected['fold_id'].values
            
        for i in range(self.n_splits):
            indices = np.arange(len(fold))
            train_index = indices[fold['fold_id'] != i]
            valid_index = indices[fold['fold_id'] == i]
            yield train_index, valid_index

# ===============
# NN
# ===============
class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with
        return_sequences = True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def se_block(input, channels, r=8):
    x = Dense(channels//r, activation="relu")(input)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])


def get_model(max_len, embedding_dim, emb_n=4, emb_n_imp=16, dout=.4, weight_decay=0.1):
    inp_cats = []
    embs = []
    for c in categorical_features_1000:
        inp_cat = Input(shape=[1], name=c)
        inp_cats.append(inp_cat)
        embs.append((Embedding(X_train[c].max() + 1, emb_n)(inp_cat)))
    for c in important_categorical:
        inp_cat = Input(shape=[1], name=c)
        inp_cats.append(inp_cat)
        embs.append((Embedding(X_train[c].max() + 1, emb_n_imp)(inp_cat)))
    cats = Flatten()(concatenate(embs))
    cats = Dense(8, activation="linear")(cats)
    cats = BatchNormalization()(cats)
    cats = PReLU()(cats)
    cats = Dropout(dout / 2)(cats)

    inp_numerical = Input(shape=(len(numerical_1000),), name="numerical")
    inp_important_numerical = Input(shape=(len(important_numerical),), name="important_numerical")
    nums = concatenate([inp_numerical, inp_important_numerical])
    nums = Dense(32, activation="linear")(nums)
    nums = BatchNormalization()(nums)
    nums = PReLU()(nums)
    nums = Dropout(dout)(nums)

    inp_img = Input(shape=(len(img_cols),), name="img")
    x_img = Dense(32, activation="linear")(inp_img)
    x_img = BatchNormalization()(x_img)
    x_img = PReLU()(x_img)
    x_img = Dropout(dout)(x_img)

    inp_desc = Input(shape=(max_len,), name="description")
    emb_desc = Embedding(len(embedding_matrix), embedding_dim, weights=[embedding_matrix], trainable=False)(inp_desc)
    emb_desc = SpatialDropout1D(0.2)(emb_desc)
    x1 = Bidirectional(CuDNNLSTM(64, return_sequences=True))(emb_desc)
    x2 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    conc = Concatenate()([max_pool2, avg_pool2])
    conc = BatchNormalization()(conc)

    conc = Dense(32, activation="linear")(conc)
    conc = BatchNormalization()(conc)
    conc = PReLU()(conc)
    conc = Dropout(dout)(conc)

    x = concatenate([conc, x_img, nums, cats, inp_important_numerical])
    x = Dropout(dout / 2)(x)

    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp_cats + [inp_numerical, inp_important_numerical, inp_img, inp_desc], outputs=out)
    model.compile(optimizer="adam", loss=rmse)
    return model


def get_model_selftrain(max_len, embedding_dim, emb_n=4, emb_n_imp=16, dout=.5, weight_decay=0.1):
    inp_cats = []
    embs = []
    for c in categorical_features:
        inp_cat = Input(shape=[1], name=c)
        inp_cats.append(inp_cat)
        embs.append((Embedding(X_train[c].max() + 1, emb_n)(inp_cat)))
    for c in important_categorical:
        inp_cat = Input(shape=[1], name=c)
        inp_cats.append(inp_cat)
        embs.append((Embedding(X_train[c].max() + 1, emb_n_imp)(inp_cat)))
    cats = Flatten()(concatenate(embs))
    imp_cats = Flatten()(concatenate(embs))
    cats = Dense(8, activation="linear")(cats)
    cats = BatchNormalization()(cats)
    cats = PReLU()(cats)
    cats = Dropout(dout / 2)(cats)

    inp_numerical = Input(shape=(len(numerical),), name="numerical")
    inp_important_numerical = Input(shape=(len(important_numerical),), name="important_numerical")
    nums = concatenate([inp_numerical, inp_important_numerical])
    nums = Dense(32, activation="linear")(nums)
    nums = BatchNormalization()(nums)
    nums = PReLU()(nums)
    nums = Dropout(dout)(nums)

    inp_dense = Input(shape=(len(dense_cols),), name="dense_cols")
    x_dense = Dense(32, activation="linear")(inp_dense)
    x_dense = BatchNormalization()(x_dense)
    x_dense = PReLU()(x_dense)

    inp_inception = Input(shape=(len(inception_cols),), name="inception_cols")
    x_inception = Dense(32, activation="linear")(inp_inception)
    x_inception = BatchNormalization()(x_inception)
    x_inception = PReLU()(x_inception)

    x_img = concatenate([x_dense, x_inception])
    x_img = Dense(32, activation="linear")(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = PReLU()(x_img)
    x_img = Dropout(dout)(x_img)

    inp_desc = Input(shape=(max_len,), name="description")
    emb_desc = Embedding(len(embedding_matrix_self), embedding_dim, weights=[embedding_matrix_self],
                         trainable=False)(inp_desc)
    emb_desc = SpatialDropout1D(0.2)(emb_desc)
    x1 = Bidirectional(CuDNNLSTM(32, return_sequences=True))(emb_desc)
    x2 = Bidirectional(CuDNNGRU(32, return_sequences=True))(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)
    avg_pool2 = GlobalAveragePooling1D()(x2)
    att2 = Attention(max_len)(x2)
    conc = Concatenate()([max_pool2, avg_pool2, att2])
    conc = se_block(conc, 64 + 64 + 64)
    conc = BatchNormalization()(conc)

    conc = Dense(32, activation="linear")(conc)
    conc = BatchNormalization()(conc)
    conc = PReLU()(conc)
    conc = Dropout(dout)(conc)

    x = concatenate([conc, x_img, nums, cats, inp_important_numerical])
    x = se_block(x, 32 + 32 + 32 + 8 + len(important_numerical))
    x = BatchNormalization()(x)
    x = Dropout(dout)(x)
    x = concatenate([x, inp_important_numerical])
    x = BatchNormalization()(x)
    x = Dropout(dout / 2)(x)

    out = Dense(1, activation="linear")(x)

    model = Model(inputs=inp_cats + [inp_numerical, inp_important_numerical, inp_dense, inp_inception, inp_desc],
                  outputs=out)
    model.compile(optimizer="adam", loss=rmse)
    return model

def get_keras_data(df, description_embeds):
    X = {
        "numerical": df[numerical_1000].values,
        "important_numerical": df[important_numerical].values,
        "description": description_embeds,
        "img": df[img_cols]
    }
    for c in categorical_features_1000 + important_categorical:
        X[c] = df[c]
    return X

def get_keras_data_selftrain(df, description_embeds):
    X = {
        "numerical": df[numerical].values,
        "important_numerical": df[important_numerical].values,
        "description": description_embeds,
        "dense_cols": df[dense_cols],
        "inception_cols": df[inception_cols]
    }
    for c in categorical_features + important_categorical:
        X[c] = df[c]
    return X

def rmse(y, y_pred):
    return K.sqrt(K.mean(K.square(y - y_pred), axis=-1))

def w2v_fornn(train_text, model, max_len):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_text))
    train_text = tokenizer.texts_to_sequences(train_text)
    train_text = pad_sequences(train_text, maxlen=max_len)
    word_index = tokenizer.word_index

    embedding_dim = model.dim
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in model:  # 0.9906
            embedding_matrix[i] = model.query(word)
            continue
        word_ = word.upper()
        if word_ in model:  # 0.9909
            embedding_matrix[i] = model.query(word_)
            continue
        word_ = word.capitalize()
        if word_ in model:  # 0.9925
            embedding_matrix[i] = model.query(word_)
            continue
        word_ = ps.stem(word)
        if word_ in model:  # 0.9927
            embedding_matrix[i] = model.query(word_)
            continue
        word_ = lc.stem(word)
        if word_ in model:  # 0.9932
            embedding_matrix[i] = model.query(word_)
            continue
        word_ = sb.stem(word)
        if word_ in model:  # 0.9933
            embedding_matrix[i] = model.query(word_)
            continue
        embedding_matrix[i] = model.query(word)

    return train_text, embedding_matrix, embedding_dim, word_index



def self_train_w2v_tonn(train_text, max_len, w2v_params, mode="w2v"):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    if mode == "w2v":
        model = word2vec.Word2Vec(train_corpus, **w2v_params)
    elif mode == "fasttext":
        model = FastText(train_corpus, **w2v_params)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_text))
    train_text = tokenizer.texts_to_sequences(train_text)
    train_text = pad_sequences(train_text, maxlen=max_len)
    word_index = tokenizer.word_index

    embedding_dim = model.vector_size
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in model:  # 0.9906
            embedding_matrix[i] = model.wv[word]
            continue
        word_ = word.upper()
        if word_ in model:  # 0.9909
            embedding_matrix[i] = model.wv[word_]
            continue
        word_ = word.capitalize()
        if word_ in model:  # 0.9925
            embedding_matrix[i] = model.wv[word_]
            continue
        word_ = ps.stem(word)
        if word_ in model:  # 0.9927
            embedding_matrix[i] = model.wv[word_]
            continue
        word_ = lc.stem(word)
        if word_ in model:  # 0.9932
            embedding_matrix[i] = model.wv[word_]
            continue
        word_ = sb.stem(word)
        if word_ in model:  # 0.9933
            embedding_matrix[i] = model.wv[word_]
            continue
        embedding_matrix[i] = np.zeros(embedding_dim)

    return train_text, embedding_matrix, embedding_dim, word_index


if __name__ == '__main__':
    init_logger()
    seed_everything(seed=seed)
    t_cols, k_cols, g_cols = [], [], []
    
    # load
    train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
    train = pd.concat([train, test], sort=True)
    train[['Description', 'Name']] = train[['Description', 'Name']].astype(str)
    train["Description_Emb"] = [analyzer_embed(text) for text in train["Description"]]
    train["Description_bow"] = [analyzer_bow(text) for text in train["Description"]]
    train['fix_Breed1'] = train['Breed1']
    train['fix_Breed2'] = train['Breed2']
    train.loc[train['Breed1'] == 0, 'fix_Breed1'] = train[train['Breed1'] == 0]['Breed2']
    train.loc[train['Breed1'] == 0, 'fix_Breed2'] = train[train['Breed1'] == 0]['Breed1']
    train['Breed1_equals_Breed2'] = (train['Breed1'] == train['Breed2']).astype(int)
    train['single_Breed'] = (train['Breed1'] * train['Breed2'] == 0).astype(int)
    train.drop(["Breed1", "Breed2"], axis=1)
    train.rename(columns={"fix_Breed1": "Breed1", "fix_Breed2": "Breed2"})
    logger.info(f'DataFrame shape: {train.shape}')
    
    with timer('common features1'): 
        with timer('merge additional state files'):
            train = merge_state_info(train)
            
        common_cols = list(train.columns)
        
        with timer('merge additional breed rating files'): 
            orig_cols = list(train.columns)
            train = merge_breed_name_sub(train)
            t_cols += [c for c in train.columns if c not in orig_cols]
            k_cols += [c for c in train.columns if c not in orig_cols]
            
            orig_cols = list(train.columns)
            train = merge_breed_name(train)
            g_cols += [c for c in train.columns if c not in orig_cols and "_main_breed_all" in c] + ["Type_second_breed"]
            
        with timer('preprocess category features'): 
            train = to_category(train, cat=categorical_features)
            
    if G_flag:
        with timer('gege features'): 
            orig_cols = train.columns

            with timer('frequency encoding'):
                freq_cols = ['RescuerID', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3', 'State']
                train = freq_encoding(train, freq_cols)

            g_cols += [c for c in train.columns if c not in orig_cols]
            
    if K_flag:
        with timer('kaeru features'): 
            orig_cols = train.columns

            with timer('frequency encoding'):
                freq_cols = ['BreedName_main_breed', 'BreedName_second_breed']
                train = freq_encoding(train, freq_cols)

            k_cols += [c for c in train.columns if c not in orig_cols if c not in kaeru_drop_cols]
            
    if T_flag:
        with timer('takuoko features'): 
            orig_cols = train.columns

            with timer('aggregation'):
                stats = ['mean', 'sum', 'median', 'min', 'max', 'var']
                groupby_dict = [
                    {
                        'key': ['Name'], 
                        'var': ['Age'], 
                        'agg': ['count']
                    },
                    {
                        'key': ['RescuerID'], 
                        'var': ['Age'], 
                        'agg': ['count']
                    },
                    {
                        'key': ['RescuerID', 'State'], 
                        'var': ['Age'], 
                        'agg': ['count']
                    },
                    {
                        'key': ['RescuerID', 'Type'], 
                        'var': ['Age'], 
                        'agg': ['count']
                    },
                    {
                        'key': ['RescuerID'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['RescuerID', 'State'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['RescuerID', 'Type'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['Type', 'Breed1', 'Breed2'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['Type', 'Breed1'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['State'], 
                        'var': ['Age', 'Quantity', 'MaturitySize', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                    {
                        'key': ['MaturitySize'], 
                        'var': ['Age', 'Quantity', 'Sterilized', 'Fee'], 
                        'agg': stats
                    },
                ]

                nunique_dict = [
                    {
                        'key': ['State'], 
                        'var': ['RescuerID'], 
                        'agg': ['nunique']
                    },
                    {
                        'key': ['Dewormed'], 
                        'var': ['RescuerID'], 
                        'agg': ['nunique']
                    },
                    {
                        'key': ['Type'], 
                        'var': ['RescuerID'], 
                        'agg': ['nunique']
                    },
                    {
                        'key': ['Type', 'Breed1'], 
                        'var': ['RescuerID'], 
                        'agg': ['nunique']
                    },
                ]

                groupby = GroupbyTransformer(param_dict=nunique_dict)
                train = groupby.transform(train)
                groupby = GroupbyTransformer(param_dict=groupby_dict)
                train = groupby.transform(train)
                diff = DiffGroupbyTransformer(param_dict=groupby_dict)
                train = diff.transform(train)
                ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
                train = ratio.transform(train)

            with timer('category embedding'):
                train[['BreedName_main_breed', 'BreedName_second_breed']] = \
                    train[['BreedName_main_breed', 'BreedName_second_breed']].astype("int32")
                for c in categorical_features:
                    train[c] = train[c].fillna(train[c].max() + 1)

                cv = CategoryVectorizer(categorical_features, n_components, 
                         vectorizer=CountVectorizer(), 
                         transformer=LatentDirichletAllocation(n_components=n_components, n_jobs=-1, learning_method='online', random_state=777),
                         name='CountLDA')
                features1 = cv.transform(train).astype(np.float32)

                cv = CategoryVectorizer(categorical_features, n_components, 
                         vectorizer=CountVectorizer(), 
                         transformer=TruncatedSVD(n_components=n_components, random_state=777),
                         name='CountSVD')
                features2 = cv.transform(train).astype(np.float32)
                train = pd.concat([train, features1, features2], axis=1)

            t_cols += [c for c in train.columns if c not in orig_cols]
            
    with timer('common features2'):      
        train[text_features].fillna('missing', inplace=True)
        with timer('preprocess metadata'): #使ってるcolsがkaeruさんとtakuokoで違う kaeruさんがfirst系は全部使うが、takuokoは使わない
            meta_parser = MetaDataParser(n_jobs=cpu_count)
            train = meta_parser.transform(train)

            k_cols += [c for c in meta_parser.g1.columns if re.match("\w*_mean_\w*mean", c)] + ["magnitude_mean", "score_mean"]
            t_cols += [c for c in meta_parser.g1.columns if re.match("\w*_sum_\w*mean", c)] + ["magnitude_mean", "score_mean"]
            g_cols  += list(meta_parser.g1.columns)

            k_cols += [c for c in meta_parser.g2.columns if ("mean_mean" in c or "mean_sum" in c or "first_first" in c) and "annots_score_normal" not in c] + \
                        ['crop_conf_first', 'crop_x_first', 'crop_y_first', 'crop_importance_first', 'crop_conf_mean', 
                       'crop_conf_sum', 'crop_importance_mean', 'crop_importance_sum']
            t_cols += [c for c in meta_parser.g2.columns if ((re.match("\w*_sum_\w*(?<!sum)$", c) and "first" not in c) \
                       or ("sum" not in c and "first" not in c)) and "annots_score_pick" not in c]
            g_cols += [c for c in meta_parser.g2.columns if "mean_mean" in c or "mean_sum" in c or "mean_var" in c and "annots_score_pick" not in c] + \
                        ['crop_conf_mean', 'crop_conf_sum', 'crop_conf_var', 'crop_importance_mean',
                       'crop_importance_sum', 'crop_importance_var']
            
        with timer('preprocess metatext'): 
            meta_features = meta_parser.metadata_files[['PetID', 'annots_top_desc', 'annots_top_desc_pick']]
            meta_features_all = meta_features.groupby('PetID')['annots_top_desc'].apply(lambda x: " ".join(x)).reset_index()
            train = train.merge(meta_features_all, how='left', on='PetID') 
            
            meta_features_pick = meta_features.groupby('PetID')['annots_top_desc_pick'].apply(lambda x: " ".join(x)).reset_index()
            train = train.merge(meta_features_pick, how='left', on='PetID') 

            sentiment_features = meta_parser.sentiment_files[['PetID', 'sentiment_text', 'sentiment_entities']]
            sentiment_features_txt = sentiment_features.groupby('PetID')['sentiment_text'].apply(lambda x: " ".join(x)).reset_index()
            train = train.merge(sentiment_features_txt, how='left', on='PetID') 
            
            sentiment_features_entities = sentiment_features.groupby('PetID')['sentiment_entities'].apply(lambda x: " ".join(x)).reset_index()
            train = train.merge(sentiment_features_entities, how='left', on='PetID') 

            train[meta_text] = train[meta_text].astype(str)
            train[meta_text].fillna("missing", inplace=True)
            del meta_features_all, meta_features_pick, meta_features, sentiment_features; gc.collect()
            
        with timer('make image features'):
            train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
            test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))
            image_files = train_image_files + test_image_files
            train_images = pd.DataFrame(image_files, columns=['image_filename'])
            train_images['PetID'] = train_images['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
        
    if T_flag:
        with timer('takuoko features'): 
            orig_cols = train.columns
            with timer('merge emoji files'):   
                train = merge_emoji(train)

            with timer('preprocess and simple features'): 
                train = get_interactions(train)

            with timer('tfidf + svd / nmf / bm25'):
                vectorizer = make_pipeline(
                    TfidfVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['tfidf_svd_{}'.format(i) for i in range(n_components)]
                                 + ['tfidf_nmf_{}'.format(i) for i in range(n_components)]
                                + ['tfidf_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('count + svd / nmf / bm25'):  
                vectorizer = make_pipeline(
                    CountVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['count_svd_{}'.format(i) for i in range(n_components)]
                                 + ['count_nmf_{}'.format(i) for i in range(n_components)]
                                + ['count_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('tfidf2 + svd / nmf / bm25'):  
                vectorizer = make_pipeline(
                    TfidfVectorizer(min_df=2,  max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['tfidf2_svd_{}'.format(i) for i in range(n_components)]
                                 + ['tfidf2_nmf_{}'.format(i) for i in range(n_components)]
                                + ['tfidf2_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('count2 + svd / nmf / bm25'):  
                vectorizer = make_pipeline(
                    CountVectorizer(min_df=2,  max_features=20000,
                    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                    ngram_range=(1, 3), stop_words='english'),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['count2_svd_{}'.format(i) for i in range(n_components)]
                                 + ['count2_nmf_{}'.format(i) for i in range(n_components)]
                                + ['count2_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('tfidf3 + svd / nmf / bm25'):
                vectorizer = make_pipeline(
                    TfidfVectorizer(min_df=30, max_features=50000, binary=True,
                                    strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                                    ngram_range=(3, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['tfidf3_svd_{}'.format(i) for i in range(n_components)]
                                            + ['tfidf3_nmf_{}'.format(i) for i in range(n_components)]
                                + ['tfidf3_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('count3 + svd / nmf / bm25'):
                vectorizer = make_pipeline(
                    CountVectorizer(min_df=30, max_features=50000, binary=True,
                                    strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                                    ngram_range=(3, 3), stop_words='english'),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description_bow']).astype(np.float32)
                X = pd.DataFrame(X, columns=['count3_svd_{}'.format(i) for i in range(n_components)]
                                            + ['count3_nmf_{}'.format(i) for i in range(n_components)]
                                + ['count3_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('description fasttext'):  
                embedding = '../input/quora-embedding/GoogleNews-vectors-negative300.bin'
                model = KeyedVectors.load_word2vec_format(embedding, binary=True)
                X = pretrained_w2v(train["Description_Emb"], model, name="gnvec").astype(np.float32)
                train = pd.concat([train, X], axis=1)
                del model; gc.collect()

            with timer('description glove'):  
                embedding = "../input/pymagnitude-data/glove.840B.300d.magnitude"
                model = Magnitude(embedding)
                X = w2v_pymagnitude(train["Description_Emb"], model, name="glove").astype(np.float32)
                train = pd.concat([train, X], axis=1)

            with timer('description glove for NN'):
                X_desc, embedding_matrix, embedding_dim, word_index = w2v_fornn(train["Description_Emb"], model,
                                                                                max_len)
                del model; gc.collect()

            with timer('description selftrain for NN'):
                w2v_params = {
                    "size": 300,
                    "seed": 0,
                    "min_count": 1,
                    "workers": 1
                }
                X_desc_self, embedding_matrix_self, embedding_dim_self, word_index_self = self_train_w2v_tonn(
                                                                     train["Description_bow"],  max_len, w2v_params)
                
            with timer('meta text bow/tfidf->svd / nmf / bm25'): 
                train['desc'] = ''
                for c in ['BreedName_main_breed', 'BreedName_second_breed', 'annots_top_desc', 'sentiment_text']:
                    train['desc'] += ' ' + train[c].astype(str)

                train["desc"] = [analyzer_bow(text) for text in train["desc"]]

                vectorizer = make_pipeline(
                    TfidfVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['desc']).astype(np.float32)
                X = pd.DataFrame(X, columns=['meta_desc_tfidf_svd_{}'.format(i) for i in range(n_components)]
                                 + ['meta_desc_tfidf_nmf_{}'.format(i) for i in range(n_components)]
                                + ['meta_desc_tfidf_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)

                vectorizer = make_pipeline(
                    CountVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=seed),
                        NMF(n_components=n_components, random_state=seed),
                        make_pipeline(
                            BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                            TruncatedSVD(n_components=n_components, random_state=seed)
                        ),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['desc']).astype(np.float32)
                X = pd.DataFrame(X, columns=['meta_desc_count_svd_{}'.format(i) for i in range(n_components)]
                                 + ['meta_desc_count_nmf_{}'.format(i) for i in range(n_components)]
                                + ['meta_desc_count_bm25_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                train.drop(['desc'], axis=1, inplace=True)
                
            with timer('image features'): 
                train['num_images'] = train['PhotoAmt']
                train['num_images_per_pet'] = train['num_images'] / train['Quantity']

            t_cols += [c for c in train.columns if c not in orig_cols]
        
    if K_flag or G_flag:
        with timer('kaeru and gege features'):
            with timer('text stats features'): 
                train = get_text_features(train)
            k_cols += ['Length_Description', 'Length_annots_top_desc', 'Lengths_sentiment_text']
            g_cols += ['Length_Description', 'Length_annots_top_desc', 'Lengths_sentiment_entities']
        
    if K_flag:
        with timer('kaeru features'): 
            orig_cols = train.columns
            with timer('enginerring age'): 
                train = get_age_feats(train)

            with timer('tfidf + svd / nmf'):
                vectorizer = make_pipeline(
                    TfidfVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=kaeru_seed),
                        # NMF(n_components=n_components, random_state=kaeru_seed),
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['Description']).astype(np.float32)
                X = pd.DataFrame(X, columns=['tfidf_k_svd_{}'.format(i) for i in range(n_components)])
                #                 + ['tfidf_k_nmf_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('description doc2vec'):  
                d2v_param = {
                    "features_num": 300,
                    "min_word_count": 10,
                    "context": 5,
                    "downsampling": 1e-3,
                    "epoch_num": 10
                }
                X = doc2vec(train["Description"], d2v_param).astype(np.float32)
                train = pd.concat([train, X], axis=1)

            with timer('annots_top_desc + svd / nmf'):
                vectorizer = make_pipeline(
                    TfidfVectorizer(),
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=kaeru_seed),
                        # NMF(n_components=n_components, random_state=kaeru_seed),
                        # n_jobs=2,
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(train['annots_top_desc_pick']).astype(np.float32)
                X = pd.DataFrame(X, columns=['annots_top_desc_k_svd_{}'.format(i) for i in range(n_components)])
                #                 + ['annots_top_desc_k_nmf_{}'.format(i) for i in range(n_components)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('aggregation'):
                stats = ['mean', 'sum', 'min', 'max']
                var = ['Age_k', 'MaturitySize_k', 'FurLength_k', 'Fee_k', 'Health_k']
                for c in ['Age', 'MaturitySize', 'FurLength', 'Fee', 'Health']:
                    train[c+"_k"] = train[c]
                groupby_dict = [
                    {
                        'key': ['RescuerID'], 
                        'var': ['Age_k'], 
                        'agg': ['count']
                    },
                    {
                        'key': ['RescuerID'], 
                        'var': ['Age_k', 'Length_Description', 'Length_annots_top_desc', 'Lengths_sentiment_text'], 
                        'agg': stats + ["var"]
                    },
                    {
                        'key': ['RescuerID'], 
                        'var': ['MaturitySize_k', 'FurLength_k', 'Fee_k', 'Health_k'], 
                        'agg': stats
                    }
                ]

                groupby = GroupbyTransformer(param_dict=groupby_dict)
                train = groupby.transform(train)
                train.drop(var, axis=1, inplace=True)

            k_cols += [c for c in train.columns if c not in orig_cols if c not in kaeru_drop_cols]
        
    if G_flag:
        with timer('gege features'): 
            orig_cols = train.columns

            with timer('tfidf + svd'):
                vectorizer = make_pipeline(
                   TfidfVectorizer(min_df=2,  max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1),
                    TruncatedSVD(n_components=n_components_gege_txt, random_state=kaeru_seed)
                )
                X = vectorizer.fit_transform(train['Description']).astype(np.float32)
                X = pd.DataFrame(X, columns=['tfidf_g_svd_{}'.format(i) for i in range(n_components_gege_txt)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('annots tfidf + svd'):
                vectorizer = make_pipeline(
                   TfidfVectorizer(min_df=2,  max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1),
                    TruncatedSVD(n_components=n_components_gege_txt, random_state=kaeru_seed)
                )
                X = vectorizer.fit_transform(train['annots_top_desc']).astype(np.float32)
                X = pd.DataFrame(X, columns=['annots_top_desc_tfidf_g_svd_{}'.format(i) for i in range(n_components_gege_txt)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('sentiment entities tfidf + svd'):
                vectorizer = make_pipeline(
                   TfidfVectorizer(min_df=2,  max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1),
                    TruncatedSVD(n_components=n_components_gege_txt, random_state=kaeru_seed)
                )
                X = vectorizer.fit_transform(train['sentiment_entities']).astype(np.float32)
                X = pd.DataFrame(X, columns=['sentiment_entities_tfidf_g_svd_{}'.format(i) for i in range(n_components_gege_txt)])
                train = pd.concat([train, X], axis=1)
                del vectorizer; gc.collect()

            with timer('image basic features'):
                train_images['image_size'] = train_images['image_filename'].apply(getSize)
                train_images['temp_size'] = train_images['image_filename'].apply(getDimensions)
                train_images['width'] = train_images['temp_size'].apply(lambda x : x[0])
                train_images['height'] = train_images['temp_size'].apply(lambda x : x[1])
                train_images = train_images.drop(['temp_size'], axis=1)

                aggs = {
                    'image_size': ['sum', 'mean', 'var'],
                    'width': ['sum', 'mean', 'var'],
                    'height': ['sum', 'mean', 'var'],
                }

                gp = train_images.groupby('PetID').agg(aggs)
                new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
                gp.columns = new_columns
                train = train.merge(gp.reset_index(), how="left", on="PetID")

            g_cols += [c for c in train.columns if c not in orig_cols]
            
    dtype_cols = ['BreedName_main_breed', 'BreedName_second_breed', 'BreedName_main_breed_all']
    train[dtype_cols] = train[dtype_cols].astype("int32")
        
    with timer('preprocess pretrained CNN'):
        if debug:
            import feather

            X = feather.read_dataframe("feature/dense121_2_X.feather")
            gp_img = X.groupby("PetID").mean().reset_index()
            train = pd.merge(train, gp_img, how="left", on="PetID")
            gp_dense_first = X.groupby("PetID").first().reset_index()

            X = feather.read_dataframe("feature/inception_resnet.feather")
            train = pd.concat((train, X), axis=1)
            t_cols += list(gp_img.columns.drop("PetID"))
            t_cols += list(X.columns.drop("PetID"))
            del gp_img;
            gc.collect()
        else:
            pet_ids = train_images['PetID'].values
            img_pathes = train_images['image_filename'].values
            # n_batches = len(pet_ids) // batch_size + 1
            n_batches = len(pet_ids) // batch_size + (len(pet_ids) % batch_size != 0)

            inp = Input((256, 256, 3))
            backbone = DenseNet121(input_tensor=inp,
                                    weights='../input/petfinderdatasets/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    include_top=False)
            x = backbone.output
            x = GlobalAveragePooling2D()(x)
            x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
            x = AveragePooling1D(4)(x)
            out = Lambda(lambda x: x[:, :, 0])(x)
            model_dense = Model(inp, out)

            inp2 = Input((256, 256, 3))
            backbone2 = InceptionResNetV2(input_tensor=inp2,
                                            weights='../input/petfinderdatasets/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                            include_top=False)
            x2 = backbone2.output
            x2 = GlobalAveragePooling2D()(x2)
            x2 = Lambda(lambda x: K.expand_dims(x, axis=-1))(x2)
            x2 = AveragePooling1D(4)(x2)
            out2 = Lambda(lambda x: x[:, :, 0])(x2)
            model_inception = Model(inp2, out2)

            features_dense = []
            features_inception = []
            for b in range(n_batches):
                start = b * batch_size
                end = (b + 1) * batch_size
                batch_pets = pet_ids[start: end]
                batch_path = img_pathes[start: end]
                batch_images_dense = np.zeros((len(batch_pets), img_size, img_size, 3))
                batch_images_incep = np.zeros((len(batch_pets), img_size, img_size, 3))
                for i, (pet_id, path) in enumerate(zip(batch_pets, batch_path)):
                    try:
                        batch_images_dense[i], batch_images_incep[i] = load_image(path)
                    except:
                        pass
                batch_preds_dense = model_dense.predict(batch_images_dense)
                batch_preds_inception = model_inception.predict(batch_images_incep)
                for i, pet_id in enumerate(batch_pets):
                    features_dense.append([pet_id] + list(batch_preds_dense[i]))
                    features_inception.append([pet_id] + list(batch_preds_inception[i]))

            X = pd.DataFrame(features_dense, columns=["PetID"] + ["dense121_2_{}".format(i) for i in
                                                                    range(batch_preds_dense.shape[1])])
            gp_img = X.groupby("PetID").mean().reset_index()
            train = pd.merge(train, gp_img, how="left", on="PetID")
            gp_dense_first = X.groupby("PetID").first().reset_index()
            t_cols += list(gp_img.columns.drop("PetID"))
            del model_dense, model_inception, X, gp_img;
            gc.collect();
            K.clear_session()

            X = pd.DataFrame(features_inception, columns=["PetID"] + ["inception_resnet_{}".format(i) for i in
                                                                        range(batch_preds_inception.shape[1])])
            gp_img = X.groupby("PetID").mean().reset_index()
            train = pd.merge(train, gp_img, how="left", on="PetID")
            t_cols += list(gp_img.columns.drop("PetID"))
            del gp_img, X;
            gc.collect()
            
    if K_flag:
        with timer('kaeru features'): 
            orig_cols = train.columns
            with timer('densenet features'):
                vectorizer = make_pipeline(
                    make_union(
                        TruncatedSVD(n_components=n_components, random_state=kaeru_seed),
                        # NMF(n_components=n_components, random_state=kaeru_seed),
                        # n_jobs=2,
                        n_jobs=1,
                    ),
                )
                X = vectorizer.fit_transform(gp_dense_first.drop(['PetID'], axis=1))
                X = pd.DataFrame(X, columns=['densenet121_svd_{}'.format(i) for i in range(n_components)])
                #                 + ['densenet121_nmf_{}'.format(i) for i in range(n_components)])
                X["PetID"] = gp_dense_first["PetID"]
                train = pd.merge(train, X, how="left", on="PetID")
                del vectorizer; gc.collect()

            k_cols += [c for c in train.columns if c not in orig_cols if c not in kaeru_drop_cols]
            
    if G_flag:
        with timer('gege features'): 
            orig_cols = train.columns
            with timer('densenet features'):
                vectorizer = TruncatedSVD(n_components=n_components_gege_img, random_state=kaeru_seed)
                X = vectorizer.fit_transform(gp_dense_first.drop(['PetID'], axis=1))
                X = pd.DataFrame(X, columns=['densenet121_g_svd_{}'.format(i) for i in range(n_components_gege_img)])
                X["PetID"] = gp_dense_first["PetID"]
                train = pd.merge(train, X, how="left", on="PetID")
                del vectorizer, gp_dense_first; gc.collect()
            g_cols += [c for c in train.columns if c not in orig_cols]
            
    logger.info(train.head())
    
    train.to_feather("all_data.feather")
    np.save("common_cols.npy", np.array(common_cols))
    np.save("t_cols.npy", np.array(t_cols))
    np.save("k_cols.npy", np.array(k_cols))
    np.save("g_cols.npy", np.array(g_cols))

    if T_flag:
        with timer('takuoko feature info'):
            categorical_features_t = list(set(categorical_features) - set(remove))
            predictors = list(set(common_cols+t_cols+categorical_features_t) - set([target] + remove))
            predictors = [c for c in predictors if c in use_cols]
            categorical_features_t = [c for c in categorical_features_t if c in predictors]
            logger.info(f'predictors / use_cols = {len(predictors)} / {len(use_cols)}')
            
            train = train.loc[:, ~train.columns.duplicated()]

            X = train.loc[:, predictors]
            y = train.loc[:, target]
            rescuer_id = train.loc[:, 'RescuerID'].iloc[:len_train]
            X_test = X[len_train:]
            X = X[:len_train]
            y = y[:len_train]
            X.to_feather("X_train_t.feather")
            X_test.reset_index(drop=True).to_feather("X_test_t.feather")

        with timer('takuoko lgbm modeling'):
            y_pred_t = np.empty(len_train,)
            y_test_t = []
            train_losses, valid_losses = [], []

            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)): 
                X_train = X.loc[train_index, :]
                X_valid = X.loc[valid_index, :]
                y_train = y[train_index]
                y_valid = y[valid_index]

                pred_val, pred_test, train_rmse, valid_rmse = run_model(X_train, y_train, X_valid, y_valid, X_test,
                                 categorical_features_t, predictors, maxvalue_dict, fold_id, MODEL_PARAMS, MODEL_NAME+"_t")
                y_pred_t[valid_index] = pred_val 
                y_test_t.append(pred_test)
                train_losses.append(train_rmse)
                valid_losses.append(valid_rmse)

            y_test_t = np.mean(y_test_t, axis=0)
            logger.info(f'train RMSE = {np.mean(train_losses)}')
            logger.info(f'valid RMSE = {np.mean(valid_losses)}')
                                    
        np.save("y_test_t.npy",y_test_t)
        np.save("y_oof_t.npy", y_pred_t)

        with timer('takuoko feature info for NN'):
            use_cols = [c for c in use_colsnn if c in train.columns]

            dense_cols = [c for c in X_train.columns if "dense" in c and "svd" not in c and "nmf" not in c]
            inception_cols = [c for c in X_train.columns if "inception" in c and "svd" not in c and "nmf" not in c]
            img_cols = dense_cols + inception_cols
            numerical = [c for c in use_cols if c not in categorical_features and c not in img_cols]
            important_numerical = [c for c in numerical if c in use_cols[:n_important]]
            numerical = [c for c in numerical if c not in use_cols[:n_important]]
            important_categorical = [c for c in categorical_features if c in use_cols[:n_important]]
            categorical_features = [c for c in categorical_features if c not in use_cols[:n_important]]

            numerical_1000 = [c for c in use_cols[n_important:1000] if c not in categorical_features and c not in img_cols]
            categorical_features_1000 = [c for c in categorical_features if c not in use_cols[n_important:1000]]

            train_cp = train[img_cols+numerical+important_numerical+categorical_features+important_categorical].copy()
            for c in categorical_features + important_categorical:
                train_cp[c] = LabelEncoder().fit_transform(train_cp[c])
            train_cp.replace(np.inf, np.nan, inplace=True)
            train_cp.replace(-np.inf, np.nan, inplace=True)
            train_cp[important_numerical + numerical] = StandardScaler().fit_transform(
                train_cp[important_numerical + numerical].rank())
            train_cp.fillna(0, inplace=True)

            X_test = train_cp.iloc[len_train:]
            X_train = train_cp.iloc[:len_train]
            X_desc_test = X_desc[len_train:]
            X_desc_train = X_desc[:len_train]
            X_desc_self_test = X_desc_self[len_train:]
            X_desc_self_train = X_desc_self[:len_train]
            del X_desc, X_desc_self; gc.collect()

        with timer('takuoko NN modeling'):
            y_pred_t_nn1 = np.empty(len_train, )
            y_test_t_nn1 = []
            train_losses, valid_losses = [], []
            x_test = get_keras_data(X_test, X_desc_test)

            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)):
                x_train = get_keras_data(X_train.iloc[train_index], X_desc_train[train_index])
                x_valid = get_keras_data(X_train.iloc[valid_index], X_desc_train[valid_index])
                y_train, y_valid = y[train_index], y[valid_index]

                model = get_model(max_len, embedding_dim)
                clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=len(X_train) // batch_size_nn, mode="triangular2")
                ckpt = ModelCheckpoint('model.hdf5', save_best_only=True,
                                       monitor='val_loss', mode='min')
                history = model.fit(x_train, y_train, batch_size=batch_size_nn, validation_data=(x_valid, y_valid),
                                    epochs=20, callbacks=[ckpt, clr_tri], verbose=2)
                model.load_weights('model.hdf5')

                y_pred_train = model.predict(x_train, batch_size=1000).reshape(-1, )
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

                y_pred_valid = model.predict(x_valid, batch_size=1000).reshape(-1, )
                valid_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
                y_pred_valid = rankdata(y_pred_valid) / len(y_pred_valid)
                y_pred_t_nn1[valid_index] = y_pred_valid

                pred_test = model.predict(x_test, batch_size=1000).reshape(-1, )
                pred_test = rankdata(pred_test) / len(pred_test)
                y_test_t_nn1.append(pred_test)

                train_losses.append(train_rmse)
                valid_losses.append(valid_rmse)

                del model;
                K.clear_session()

            y_test_t_nn1 = np.mean(y_test_t_nn1, axis=0)
            logger.info(f'train RMSE = {np.mean(train_losses)}')
            logger.info(f'valid RMSE = {np.mean(valid_losses)}')

        np.save("y_test_t_nn1.npy", y_test_t_nn1)
        np.save("y_oof_t_nn1.npy", y_pred_t_nn1)

        with timer('takuoko selftrained NN modeling'):
            y_pred_t_nn2 = np.empty(len_train, )
            y_test_t_nn2 = []
            train_losses, valid_losses = [], []
            x_test = get_keras_data_selftrain(X_test, X_desc_self_test)

            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)):
                x_train = get_keras_data_selftrain(X_train.iloc[train_index], X_desc_self_train[train_index])
                x_valid = get_keras_data_selftrain(X_train.iloc[valid_index], X_desc_self_train[valid_index])
                y_train, y_valid = y[train_index], y[valid_index]

                model = get_model_selftrain(max_len, embedding_dim_self)
                clr_tri = CyclicLR(base_lr=1e-5, max_lr=1e-2, step_size=len(X_train) // batch_size_nn, mode="triangular2")
                ckpt = ModelCheckpoint('model.hdf5', save_best_only=True,
                                       monitor='val_loss', mode='min')
                history = model.fit(x_train, y_train, batch_size=batch_size_nn, validation_data=(x_valid, y_valid),
                                    epochs=20, callbacks=[ckpt, clr_tri], verbose=2)
                model.load_weights('model.hdf5')

                y_pred_train = model.predict(x_train, batch_size=1000).reshape(-1, )
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

                y_pred_valid = model.predict(x_valid, batch_size=1000).reshape(-1, )
                valid_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
                y_pred_valid = rankdata(y_pred_valid) / len(y_pred_valid)
                y_pred_t_nn2[valid_index] = y_pred_valid

                pred_test = model.predict(x_test, batch_size=1000).reshape(-1, )
                pred_test = rankdata(pred_test) / len(pred_test)
                y_test_t_nn2.append(pred_test)

                train_losses.append(train_rmse)
                valid_losses.append(valid_rmse)

                del model;
                K.clear_session()

            y_test_t_nn2 = np.mean(y_test_t_nn2, axis=0)
            logger.info(f'train RMSE = {np.mean(train_losses)}')
            logger.info(f'valid RMSE = {np.mean(valid_losses)}')

        np.save("y_test_t_nn2.npy", y_test_t_nn2)
        np.save("y_oof_t_nn2.npy", y_pred_t_nn2)

        y_test_t_nn = (y_test_t_nn1 + y_test_t_nn2) / 2
        y_pred_t_nn = (y_pred_t_nn1 + y_pred_t_nn2) / 2
    
    if K_flag:
        with timer('kaeru feature info'):
            kaeru_cat_cols = None
            predictors = list(set(common_cols+k_cols) - set([target] + remove+kaeru_drop_cols))

            X = train.loc[:, predictors]
            y = train.loc[:, target]
            rescuer_id = train.loc[:, 'RescuerID'].iloc[:len_train]
            X_test = X[len_train:]
            X = X[:len_train]
            y = y[:len_train]
            X.to_feather("X_train_k.feather")
            X_test.reset_index(drop=True).to_feather("X_test_k.feather")

        with timer('kaeru modeling'):
            y_pred_k = np.empty(len_train,)
            y_test_k = []
            train_losses, valid_losses = [], []

            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)):
                X_train = X.loc[train_index, :]
                X_valid = X.loc[valid_index, :]
                y_train = y[train_index]
                y_valid = y[valid_index]

                pred_val, pred_test, train_rmse, valid_rmse = run_model(X_train, y_train, X_valid, y_valid, X_test,
                                 kaeru_cat_cols, predictors, maxvalue_dict, fold_id, KAERU_PARAMS, MODEL_NAME+"_k")
                y_pred_k[valid_index] = pred_val 
                y_test_k.append(pred_test)
                train_losses.append(train_rmse)
                valid_losses.append(valid_rmse)

            y_test_k = np.mean(y_test_k, axis=0)
            logger.info(f'train RMSE = {np.mean(train_losses)}')
            logger.info(f'valid RMSE = {np.mean(valid_losses)}')

        np.save("y_test_k.npy",y_test_k)
        np.save("y_oof_k.npy", y_pred_k)
        
    if G_flag:
        with timer('gege feature info'):
            predictors = list(set(common_cols+g_cols) - set([target] + remove+gege_drop_cols))
            categorical_features_g = [c for c in categorical_features if c in predictors]

            X = train.loc[:, predictors]
            y = train.loc[:, target]
            rescuer_id = train.loc[:, 'RescuerID'].iloc[:len_train]
            X_test = X[len_train:]
            X = X[:len_train]
            y = y[:len_train]
            X.to_feather("X_train_g.feather")
            X_test.reset_index(drop=True).to_feather("X_test_g.feather")
            
        with timer('gege adversarial validation'):
            train_idx = range(0, len_train)
            X_adv = train.loc[:, predictors]
            y_adv = np.array([0 for i in range(len(X))] + [1 for i in range(len(X_test))])
            
            X_adv_tr, X_adv_tst, y_adv_tr, y_adv_tst = train_test_split(X_adv,y_adv, test_size=0.20, shuffle = True, random_state = 42)
            
            lgtrain = lgb.Dataset(X_adv_tr, y_adv_tr,
                                  categorical_feature=categorical_features_g, 
                                  feature_name=predictors)
            lgvalid = lgb.Dataset(X_adv_tst, y_adv_tst,
                                  categorical_feature=categorical_features_g, 
                                  feature_name=predictors)

            lgb_adv = lgb.train(
                ADV_PARAMS,
                lgtrain,
                num_boost_round=20000,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train','valid'],
                early_stopping_rounds=500,
                verbose_eval=20000
            )

            train_preds = lgb_adv.predict(X_adv.iloc[train_idx])
            extract_idx = np.argsort(-train_preds)[:int(len(train_idx)*0.85)]
            np.save('extract_idx.npy', extract_idx)
            
            del X_adv_tr, X_adv_tst, y_adv_tr, y_adv_tst, X_adv, y_adv, lgb_adv; gc.collect()

        with timer('gege modeling'):
            X = X.iloc[extract_idx].reset_index(drop=True)
            y = y[extract_idx].reset_index(drop=True)
            rescuer_id = rescuer_id[extract_idx].reset_index(drop=True)
            y_pred_g = np.empty(len(extract_idx),)
            y_test_g = []
            train_losses, valid_losses = [], []

            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)): 
                X_train = X.loc[train_index, :]
                X_valid = X.loc[valid_index, :]
                y_train = y[train_index]
                y_valid = y[valid_index]

                pred_val, pred_test, train_rmse, valid_rmse = run_xgb_model(X_train, y_train,
                                                                            X_valid, y_valid, X_test, predictors, maxvalue_dict, 
                                                                            fold_id, MODEL_PARAMS_XGB, MODEL_NAME+"_g")
                y_pred_g[valid_index] = pred_val 
                y_test_g.append(pred_test)
                train_losses.append(train_rmse)
                valid_losses.append(valid_rmse)

            y_test_g = np.mean(y_test_g, axis=0)
            logger.info(f'train RMSE = {np.mean(train_losses)}')
            logger.info(f'valid RMSE = {np.mean(valid_losses)}')

        np.save("y_test_g.npy",y_test_g)
        np.save("y_oof_g.npy", y_pred_g)
        np.save("extract_idx.npy", extract_idx)
    
    if T_flag and K_flag and G_flag:
        with timer('stacking'):
            X = np.concatenate([y_pred_t[extract_idx].reshape(-1, 1), 
                                y_pred_t_nn[extract_idx].reshape(-1, 1), 
                                y_pred_k[extract_idx].reshape(-1, 1), 
                                y_pred_g.reshape(-1, 1), 
                                ], axis=1)
            X_test = np.concatenate([y_test_t.reshape(-1, 1), 
                                     y_test_t_nn.reshape(-1, 1), 
                                     y_test_k.reshape(-1, 1), 
                                     y_test_g.reshape(-1, 1),
                                    ], axis=1)

            y_pred_stack_ridge = np.empty([len(extract_idx), 1])
            y_test_stack_ridge = []
            cv = StratifiedGroupKFold(n_splits=n_splits)
            for fold_id, (train_index, valid_index) in enumerate(cv.split(range(len(X)), y=y, groups=rescuer_id)): 
                X_train = X[train_index]
                X_valid = X[valid_index]
                y_train = y[train_index]
                y_valid = y[valid_index]
            
                model = Ridge(alpha=0.01)
                model.fit(X_train, y_train)
                y_pred_valid = model.predict(X_valid)
                y_pred_test = model.predict(X_test)
            
                y_test_stack_ridge.append(y_pred_test)
                y_pred_stack_ridge[valid_index] = y_pred_valid.reshape(-1, 1)
            
            y_test = np.mean(y_test_stack_ridge, axis=0)
            y_test = rankdata(y_test) / len(y_test)
            y_pred = rankdata(y_pred_stack_ridge) / len(y_pred_stack_ridge)
    elif T_flag and K_flag:
        y_pred = y_pred_t*0.5 + y_pred_k*0.5
        y_test = y_test_t*0.5 + y_test_k*0.5
    elif T_flag and G_flag:
        y_pred = y_pred_t[extract_idx]*0.5 + y_pred_g*0.5
        y_test = y_test_t*0.5 + y_test_g*0.5
    elif G_flag and K_flag:
        y_pred = y_pred_g*0.5 + y_pred_k[extract_idx]*0.5
        y_test = y_test_g*0.5 + y_test_k*0.5
    elif T_flag:
        y_pred = y_pred_t
        y_test = y_test_t
    elif K_flag:
        y_pred = y_pred_k
        y_test = y_test_k
    elif G_flag:
        y_pred = y_pred_g
        y_test = y_test_g
    
    with timer('optimize threshold'):
        optR = OptimizedRounder()
        optR.fit(y_pred, y)
        coefficients = optR.coefficients()
        y_pred = optR.predict(y_pred, coefficients)
        score = get_score(y, y_pred)
        logger.info(f'Coefficients = {coefficients}')
        logger.info(f'QWK = {score}')
        y_test = optR.predict(y_test, coefficients).astype(int)
        
    with timer('postprocess'):
        try:
            submission_with_postprocessV3(y_test)
        except:
            submission(y_test)
    