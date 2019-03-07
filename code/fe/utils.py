# -*- coding: utf-8 -*-
'''
feature: v1, 2, 3, 4, 10, 11
feature: v1, 2, 3, 4, 11, 13, 14, 17, 18, 19, 22, 23
model: v10
'''
import itertools
import json
import gc
import glob
import os
import time
import cv2
import re
import nltk
nltk.download('punkt')
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from keras.applications.densenet import preprocess_input, DenseNet121, DenseNet169
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
from keras.models import Model
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union


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
FIT_PARAMS = {
    'num_boost_round': 5000,
    'early_stopping_rounds': 100,
    'verbose_eval': 100,
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
remove = ['index', 'seq_text', 'PetID', 'Name', 'Description', 'RescuerID', 'StateName', 'annots_top_desc']


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

def submission(y_pred):
    logger.info('making submission file...')
    df_sub = pd.read_csv('../input/petfinder-adoption-prediction/test/sample_submission.csv')
    df_sub[target] = y_pred
    df_sub.to_csv('submission.csv', index=False)

def analyzer(text):
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower() # 小文字化
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    text = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', text) # 記号をスペースに置き換え
    text = nltk.word_tokenize(text)
    
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text = [stemmer.stem(t) for t in text]
    
    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは除外
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 2: #  1文字、0文字（空文字）は除外
            continue
        words.append(word)
        
    return " ".join(words)
    
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
    breeds = pd.read_csv('../../input/petfinder-adoption-prediction/breed_labels.csv')
    df = pd.read_json('../../input/cat-and-dog-breeds-parameters/rating.json')
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
    emoji = pd.read_csv('../input/emoji-sentiment-data/Emoji_Sentiment_Data_v1.0.csv')
    emoji2 = pd.read_csv('../input/emoji-sentiment-data/Emojitracker_20150604.csv')
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
    
def count_nan(train):
    train['nan_count'] = train.isnull().sum(axis=1) 
    return train

def get_interactions(train):
    interaction_features = ['Age', 'Quantity']
    for (c1, c2) in combinations(interaction_features, 2):
        train[c1 + '_mul_' + c2] = train[c1] * train[c2]
        train[c1 + '_div_' + c2] = train[c1] / train[c2]
    return train

def get_text_features(train):
    train['num_chars'] = train['Description'].apply(len)
    train['num_capitals'] = train['Description'].apply(lambda x: sum(1 for c in x if c.isupper()))
    train['caps_vs_length'] = train.apply(lambda row: row['num_capitals'] / (row['num_chars']+1e-5), axis=1)
    train['num_exclamation_marks'] = train['Description'].apply(lambda x: x.count('!'))
    train['num_question_marks'] = train['Description'].apply(lambda x: x.count('?'))
    train['num_punctuation'] = train['Description'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    train['num_symbols'] = train['Description'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    train['num_words'] = train['Description'].apply(lambda x: len(x.split()))
    train['num_unique_words'] = train['Description'].apply(lambda x: len(set(w for w in x.split())))
    train['words_vs_unique'] = train['num_unique_words'] / train['num_words']
    train['num_smilies'] = train['Description'].apply(lambda x: sum(x.count(w) for w in (':-)', ':)', ';-)', ';)')))
    train['name_in_description'] = train.apply(lambda x: x['Name'] in x['Description'], axis=1).astype(int)
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
    def __init__(self):
        # sentiment files
        train_sentiment_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_sentiment/*.json'))
        test_sentiment_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_sentiment/*.json'))
        sentiment_files = train_sentiment_files + test_sentiment_files
        self.sentiment_files = pd.DataFrame(sentiment_files, columns=['sentiment_filename'])
        self.sentiment_files['PetID'] = self.sentiment_files['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
        
        # metadata files
        train_metadata_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/train_metadata/*.json'))
        test_metadata_files = sorted(glob.glob('../../input/petfinder-adoption-prediction/test_metadata/*.json'))
        metadata_files = train_metadata_files + test_metadata_files
        self.metadata_files = pd.DataFrame(metadata_files, columns=['metadata_filename'])
        self.metadata_files['PetID'] = self.metadata_files['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    
    def open_json_file(self, filename):
        with open(filename, 'r', encoding="utf-8") as f:
            metadata_file = json.load(f)
        return metadata_file
    
    def get_stats(self, array, name):
        stats = [np.mean, np.max, np.min, np.sum]
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

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()
        
        file_sentiment.update(file_sentences_sentiment)
        
        return pd.Series(file_sentiment)
    
    def parse_metadata(self, file):
        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            label_annotations = file['labelAnnotations']
            file_top_score = [x['score'] for x in label_annotations]
            file_top_desc = [x['description'] for x in label_annotations]       
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
            'n_faces': n_faces,
            'n_text_annotations': file_n_text_annotations,
            'crop_conf': file_crop_conf,
            'crop_x': file_crop_x,
            'crop_y': file_crop_y,
            'crop_importance': file_crop_importance,
        }
        metadata.update(self.get_stats(file_top_score, 'annots_score'))
        metadata.update(self.get_stats(file_color_score, 'color_score'))
        metadata.update(self.get_stats(file_color_pixelfrac, 'color_pixel_score'))
        metadata.update(self.get_stats(file_color_red, 'color_red_score'))
        metadata.update(self.get_stats(file_color_blue, 'color_blue_score'))
        metadata.update(self.get_stats(file_color_green, 'color_green_score'))
        metadata.update(self.get_stats(dog_cat_scores, 'dog_cat_scores'))
        metadata.update(self.get_stats(dog_cat_topics, 'dog_cat_topics'))
        metadata.update(self.get_stats(is_dog_or_cat, 'is_dog_or_cat'))
        metadata.update(self.get_stats(file_len_text, 'len_text'))

        return pd.Series(metadata)
    
    def _transform(self, path, sentiment=True):
        file = self.open_json_file(path)
        if sentiment:
            result = self.parse_sentiment_file(file)
        else:
            result = self.parse_metadata(file)
        return result
    
    
# ===============
# Model
# ===============
def get_score(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def get_y():
    return pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv', usecols=[target]).values.flatten()
    
def run_model(X_train, y_train, X_valid, y_valid, X_test,
            categorical_features, numerical_features,
            predictors, maxvalue_dict, fold_id):
    train = lgb.Dataset(X_train, y_train, 
                        categorical_feature=categorical_features, 
                        feature_name=predictors)
    valid = lgb.Dataset(X_valid, y_valid, 
                        categorical_feature=categorical_features, 
                        feature_name=predictors)
    evals_result = {}
    model = lgb.train(
        MODEL_PARAMS,
        train,
        valid_sets=[valid],
        valid_names=['valid'],
        evals_result=evals_result,
        **FIT_PARAMS
    )
    logger.info(f'Best Iteration: {model.best_iteration}')

    # validation score
    y_pred_valid = model.predict(X_valid)

    # feature importances
    importances = pd.DataFrame()
    importances['feature'] = predictors
    importances['gain'] = model.feature_importance(importance_type='gain')
    importances['split'] = model.feature_importance(importance_type='split')
    importances['fold'] = fold_id
    importances.to_pickle(f'feature_importances_{MODEL_NAME}_fold{fold_id}.pkl')

    # save model
    model.save_model(f'{MODEL_NAME}_fold{fold_id}.txt')

    # predict test
    y_pred_test = model.predict(X_test)

    # save predictions
    np.save(f'{MODEL_NAME}_train_fold{fold_id}.npy', y_pred_valid)
    np.save(f'{MODEL_NAME}_test_fold{fold_id}.npy', y_pred_test)

    return y_pred_test
 
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

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = get_score(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

plt.rcParams['figure.figsize'] = (12, 9)

pd.options.display.max_rows = 128
pd.options.display.max_columns = 128

train = pd.read_csv('../../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../../input/petfinder-adoption-prediction/test/test.csv')
train = pd.concat([train, test], sort=True)
train[['Description', 'Name']] = train[['Description', 'Name']].astype(str)
train["Description"] = [analyzer(text) for text in train["Description"]]
train["Name"] = [analyzer(text) for text in train["Name"]]
train['fix_Breed1'] = train['Breed1']
train['fix_Breed2'] = train['Breed2']
train.loc[train['Breed1'] == 0, 'fix_Breed1'] = train[train['Breed1'] == 0]['Breed2']
train.loc[train['Breed1'] == 0, 'fix_Breed2'] = train[train['Breed1'] == 0]['Breed1']
train['Breed1_equals_Breed2'] = (train['Breed1'] == train['Breed2']).astype(int)
train['single_Breed'] = (train['Breed1'] * train['Breed2'] == 0).astype(int)
train.drop(["Breed1", "Breed2"], axis=1)
train.rename(columns={"fix_Breed1": "Breed1", "fix_Breed2": "Breed2"})
