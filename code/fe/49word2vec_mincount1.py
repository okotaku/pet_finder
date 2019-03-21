from utils import *

import jieba, sys

nltk.download('stopwords')
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), ' ')


def preprocess_text(text,
                    filters='!"#$%&()*+,-.。 /:;<=>?@[\\]^_`{|}~\t\n',
                    split=' ',
                    lower=True,
                    remove_stopwords=False,
                    remove_non_ascii=False,
                    remove_emoji=False):
    if lower:
        text = text.lower()
    if remove_non_ascii:
        pattern = re.compile('[^(?u)\w\s]+')
        text = re.sub(pattern, '', text)
    if remove_stopwords:
        text = ' '.join([w for w in text.split() if not w in en_stopwords])
    if remove_emoji:
        text = text.translate(non_bmp_map)

    maketrans = str.maketrans
    translate_map = maketrans(filters, split * len(filters))
    text = text.translate(translate_map)
    text = text.replace("'", " ' ")

    if len(text) == 0:
        text = ' '

    # split
    kanji = re.compile('[一-龥]')
    if re.search(kanji, text):
        return [word for word in jieba.cut(text) if word != ' ']
    else:
        return text.split(split)


def w2v(train_corpus, w2v_params):
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
        "size": 300,
        "iter": 5,
        "seed": 0,
        "min_count": 1,
        "workers": 1
    }
    train["Description_Emb2"] = train["Description"].apply(preprocess_text)
    result = w2v(train["Description_Emb2"], w2v_params)
    result.to_feather("../feature/w2v_v2.feather")