import spacy
from utils import *


def spacy_d2v(train_text):
    nlp = spacy.load('en_core_web_md')
    result = np.array([nlp(text).vector for text in train["Description"].values])

    d2v_cols = ["spacy_d2v_md{}".format(i) for i in range(1, result.shape[1] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


def spacy_d2v(train_text):
    nlp = spacy.load('en_core_web_sm')
    result = np.zeros((len(train_text), 384))
    for i, text in enumerate(train["Description"].values):
        d2v = nlp(text).vector
        if len(d2v) != 0:
            result[i] = d2v

    d2v_cols = ["spacy_d2v_sm{}".format(i) for i in range(1, result.shape[1] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


def spacy_d2v(train_text):
    nlp = spacy.load('en_core_web_lg')
    result = np.array([nlp(text).vector for text in train["Description"].values])

    d2v_cols = ["spacy_d2v_lg{}".format(i) for i in range(1, result.shape[1] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


def spacy_d2v(train_text):
    nlp = spacy.load('en_vectors_web_lg')
    result = np.array([nlp(text).vector for text in train["Description"].values])

    d2v_cols = ["spacy_d2v_vlg{}".format(i) for i in range(1, result.shape[1] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


if __name__ == '__main__':
    result = spacy_d2v(train["Description"])
    result.to_feather("../feature/spacy_d2v_md.feather")

    result = spacy_d2v(train["Description"])
    result.to_feather("../feature/spacy_d2v_sm.feather")

    result = spacy_d2v(train["Description"])
    result.to_feather("../feature/spacy_d2v_lg.feather")

    result = spacy_d2v(train["Description"])
    result.to_feather("../feature/spacy_d2v_vlg.feather")
