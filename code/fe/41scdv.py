import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.mixture import GaussianMixture
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *


def scdv(train_text, embedding, clusters_num=60):
    '''
    input
    ---
    train_text(list): list of train text
    test_text(list): list of test text
    w2v_model(str): path of word2vec model
    embedding_dim(int): dimension of embedding
    fe_save_path(str): path of feature directory
    clusters_num(int): n_components of GMM
    '''
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    model = KeyedVectors.load_word2vec_format(embedding, binary=True)
    word_vectors = model.wv.vectors

    gmm = GaussianMixture(n_components=clusters_num, covariance_type='tied', max_iter=50)
    gmm.fit(word_vectors)

    tfv = TfidfVectorizer(max_features=None, ngram_range=(1, 1))
    tfv.fit(list(train_text))

    idf_dic = dict(zip(tfv.get_feature_names(), tfv._tfidf.idf_))
    assign_dic = dict(zip(model.wv.index2word, gmm.predict(word_vectors)))
    soft_assign_dic = dict(zip(model.wv.index2word, gmm.predict_proba(word_vectors)))

    word_topic_vecs = {}
    for word in assign_dic:
        word_topic_vecs[word] = np.zeros(model.vector_size * clusters_num, dtype=np.float32)
        for i in range(0, clusters_num):
            try:
                word_topic_vecs[word][i * model.vector_size:(i + 1) * model.vector_size] = model.wv[word] * soft_assign_dic[word][
                    i] * idf_dic[word]
            except:
                continue

    result_col = ["scdv{}".format(i) for i in range(1, model.vector_size*clusters_num+1)]
    scdvs_train = _get_scdv_vector(train_corpus, word_topic_vecs, clusters_num, model.vector_size)
    X = pd.DataFrame(scdvs_train)
    X.columns = result_col

    return X


def _get_scdv_vector(corpus, word_topic_vecs, clusters_num, embedding_dim, p=0.04):
    scdvs = np.zeros((len(corpus), clusters_num * embedding_dim), dtype=np.float32)

    a_min = 0
    a_max = 0

    for i, text in enumerate(corpus):
        tmp = np.zeros(clusters_num * embedding_dim, dtype=np.float32)
        for word in text:
            if word in word_topic_vecs:
                tmp += word_topic_vecs[word]
        norm = np.sqrt(np.sum(tmp ** 2))
        if norm > 0:
            tmp /= norm
        a_min += min(tmp)
        a_max += max(tmp)
        scdvs[i] = tmp

    a_min = a_min * 1.0 / len(corpus)
    a_max = a_max * 1.0 / len(corpus)
    thres = (abs(a_min) + abs(a_max)) / 2
    thres *= p

    scdvs[abs(scdvs) < thres] = 0

    return scdvs





if __name__ == '__main__':
    embedding = '../../input/quora-embedding/GoogleNews-vectors-negative300.bin'
    X = scdv(train["Description"], embedding, clusters_num=5)
