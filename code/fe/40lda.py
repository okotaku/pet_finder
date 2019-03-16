from collections import defaultdict
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing.text import text_to_word_sequence
from utils import *


def w2v(train_text, n_topics=5):
    train_corpus = [text_to_word_sequence(text) for text in train_text]
    dictionary = Dictionary(train_corpus)

    score_by_topic = defaultdict(int)
    corpus = [dictionary.doc2bow(text) for text in train_corpus]
    model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)

    lda_score = []
    for text in corpus:
        scores = []
        for topic, score in model[text]:
            scores.append(float(score))
        lda_score.append(scores)

    w2v_cols = ["lda{}".format(i) for i in range(n_topics)]
    result = pd.DataFrame(lda_score, columns=w2v_cols)

    return result


if __name__ == '__main__':
    result = w2v(train["Description"])
    result.to_feather("../feature/lda.feather")