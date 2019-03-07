from os.path import join
from pymagnitude import *
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def w2v_pymagnitude(train_text, path, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    model = Magnitude(path)

    result = []
    for text in train_corpus:
        n_skip = 0
        vec = np.zeros(model.dim)
        for n_w, word in enumerate(text):
            try:
                vec_ = model.query(word)
            except:
                n_skip += 1
                continue
            if n_w == 0:
                vec = vec_
            else:
                vec = vec + vec_
        vec = vec / (n_w - n_skip + 1)
        result.append(vec)

    w2v_cols = ["{}_mag{}".format(name, i) for i in range(1, model.dim + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result


if __name__ == '__main__':
    path = "../../input/pymagnitude_data/glove.840B.300d.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="glove")
    result.to_feather("../feature/glove_mag.feather")

    path = "../../input/pymagnitude_data/wiki-news-300d-1M-subword.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="fasttext_light")
    result.to_feather("../feature/fasttext_light_mag.feather")

    path = "../../input/pymagnitude_data/wiki-news-300d-1M-subword.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="fasttext_light")
    result.to_feather("../feature/fasttext_light_mag.feather")

    path = "../pymag/glove.6B.300d_wiki_light.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="glove_wiki_light")
    result.to_feather("../features/glove_wiki_light_mag.feather")

    path = "../pymag/glove.6B.300d_wiki_light.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="glove_wiki_light")
    result.to_feather("../features/glove_wiki_light_mag.feather")

    path = "../pymag/elmo_2x4096_512_2048cnn_2xhighway_weights_3072light.magnitude"
    result = w2v_pymagnitude(train["Description"], path, name="elmo3072_light")
    result.to_feather("../features/elmo3072_light_mag.feather")