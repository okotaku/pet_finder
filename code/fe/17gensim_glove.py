from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def w2v(train_text):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    glove_file = '../../input/glove.840B.300d/glove.840B.300d.txt'
    tmp_file = "../../input/glove.840B.300d/test_word2vec.txt"
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

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

    w2v_cols = ["glove{}".format(i) for i in range(1, 300 + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result


if __name__ == '__main__':
    result = w2v(train["Description"])
    result.to_feather("../feature/glove.feather")