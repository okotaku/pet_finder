from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def d2v(train_text, d2v_params):
    train_corpus = [TaggedDocument(words=text_to_word_sequence(text), tags=[i]) for i, text in enumerate(train_text)]

    model = Doc2Vec(train_corpus, **d2v_params)
    model.save("d2v.model")

    result = [model.infer_vector(text_to_word_sequence(text)) for text in train_text]
    d2v_cols = ["d2v{}".format(i) for i in range(1, d2v_params["size"] + 1)]
    result = pd.DataFrame(result)
    result.columns = d2v_cols

    return result


if __name__ == '__main__':
    d2v_params = {
        "size": 200,
        "window": 5,
        "max_vocab_size": 20000,
        "seed": 0,
        "min_count": 10,
        "workers": 1
    }
    result = d2v(train["Description"], d2v_params)
    result.to_feather("../feature/d2v.feather")