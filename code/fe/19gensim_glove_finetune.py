import codecs
from os.path import join
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def w2v(train_text, w2v_params):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    glove_file = '../../input/glove.840B.300d/glove.840B.300d.txt'
    tmp_file = "../../input/glove.840B.300d/test_word2vec.txt"
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    model2 = word2vec.Word2Vec(**w2v_params)
    model2.build_vocab(train_corpus)
    total_examples = model2.corpus_count
    model2.build_vocab([list(model.vocab.keys())], update=True)
    with codecs.open(tmp_file, "r", "Shift-JIS", "ignore") as file:
        model2.intersect_word2vec_format(file, binary=True, lockf=1.0)
    model2.train(train_corpus, total_examples=total_examples, epochs=model2.iter)
    with codecs.open(tmp_file, "r", "Shift-JIS", "ignore") as file:
        model2.intersect_word2vec_format(file, binary=True, lockf=1.0)

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

    w2v_cols = ["glove_finetune{}".format(i) for i in range(1, 300 + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result


if __name__ == '__main__':
    w2v_params = {
        "size": 300,
        "window": 5,
        "max_vocab_size": 20000,
        "seed": 0,
        "min_count": 10,
        "workers": 1
    }
    result = w2v(train["Description"], w2v_params)
    result.to_feather("../feature/glove_finetune.feather")