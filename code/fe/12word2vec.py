from os.path import join
from gensim.models import word2vec
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def w2v(train_text, w2v_params):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = word2vec.Word2Vec(train_corpus, **w2v_params)
    model.save("model.model")

    result = []
    for text in train_corpus:
        n_skip = 0
        for n_w, word in enumerate(text):
            try:
                vec_ = model.wv[word]
            except:
                n_skip += 1
                continue
            if n_w == 0:
                vec = vec_
            else:
                vec = vec + vec_
        vec = vec / (n_w - n_skip + 1)
        result.append(vec)

    w2v_cols = ["wv{}".format(i) for i in range(1, w2v_params["size"] + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result


if __name__ == '__main__':
    w2v_params = {
        "size": 200,
        "window": 5,
        "max_vocab_size": 20000,
        "seed": 0,
        "min_count": 10,
        "workers": 1
    }
    result = w2v(train["Description"], w2v_params)
    result.to_feather("../feature/w2v.feather")