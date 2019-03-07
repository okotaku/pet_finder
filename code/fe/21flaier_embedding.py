import flair, torch

if torch.cuda.is_available():
    flair.device = torch.device('cuda:0')
else:
    flair.device = torch.device('cpu')

from os.path import join
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from flair.embeddings import ELMoEmbeddings
from utils import *


def w2v_flair(train_text, embedding, name):
    result = []
    for text in train_text:
        if len(text) == 0:
            result.append(np.zeros(embedding.embedding_length))
            continue

        n_w = 0
        sentence = Sentence(text)
        embedding.embed(sentence)
        for token in sentence:
            vec_ = np.array(token.embedding.detach().numpy())
            if np.sum(vec_) == 0:
                continue
            if n_w == 0:
                vec = vec_
            else:
                vec = vec + vec_
            n_w += 1

        if n_w == 0: n_w = 1
        vec = vec / n_w
        result.append(vec)

    w2v_cols = ["{}{}".format(name, i) for i in range(1, embedding.embedding_length + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols

    return result


if __name__ == '__main__':
    with timer('glove'):
       embedding = WordEmbeddings('glove')
       result = w2v_flair(train["Description"], embedding, name="glove")
       result.to_feather("../feature/glove_flair.feather")

    with timer('extvec'):
        embedding = WordEmbeddings('extvec')
        result = w2v_flair(train["Description"], embedding, name="extvec")
        result.to_feather("../feature/extvec_flair.feather")

    with timer('crawl'):
        embedding = WordEmbeddings('crawl')
        result = w2v_flair(train["Description"], embedding, name="crawl")
        result.to_feather("../feature/crawl_flair.feather")

    with timer('turian'):
        embedding = WordEmbeddings('turian')
        result = w2v_flair(train["Description"], embedding, name="turian")
        result.to_feather("../feature/turian_flair.feather")

    with timer('twitter'):
        embedding = WordEmbeddings('twitter')
        result = w2v_flair(train["Description"], embedding, name="twitter")
        result.to_feather("../feature/twitter_flair.feather")

    with timer('news'):
        embedding = FlairEmbeddings('news-forward')
        result = w2v_flair(train["Description"], embedding, name="news_flair")
        result.to_feather("../feature/news_flair.feather")

    with timer('char'):
        embedding = CharacterEmbeddings()
        result = w2v_flair(train["Description"], embedding, name="char")
        result.to_feather("../feature/char_flair.feather")

    with timer('byte_pair'):
        embedding = BytePairEmbeddings('en')
        result = w2v_flair(train["Description"], embedding, name="byte_pair")
        result.to_feather("../feature/byte_pair_flair.feather")

    with timer('elmo'):
        embedding = ELMoEmbeddings('medium')
        result = w2v_flair(train["Description"], embedding, name="elmo")
        result.to_feather("../feature/elmo_flair.feather")
