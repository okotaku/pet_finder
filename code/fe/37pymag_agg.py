from os.path import join
from pymagnitude import *
from utils import *


def w2v_pymagnitude(train_text, path, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    model = Magnitude(path)

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
            vec = np.zeros((1, model.dim))
        else:
            vec = np.array(vec)
        mean_vec = np.mean(vec, axis=0)
        median_vec = np.median(vec, axis=0)
        min_vec = np.min(vec, axis=0)
        max_vec = np.max(vec, axis=0)
        var_vec = np.var(vec, axis=0)
        result.append(list(mean_vec) + list(median_vec) + list(min_vec) + list(max_vec) + list(var_vec))

    w2v_cols = ["{}_mag{}_{}".format(name, i, stats) for stats in ["mean", "median", "min", "max", "var"] for i in
                range(1, model.dim + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result

path = "../../input/pymagnitude-data/glove.840B.300d.magnitude"
result = w2v_pymagnitude(train["Description"], path, name="glove")
result.to_feather("../feature/glove_mag_agg.feather")