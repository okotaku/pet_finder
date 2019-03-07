from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from utils import *


def w2v_flair(train_text, embedding, name):
    result = []
    for text in train_text:
        if len(text) == 0:
            result.append(np.zeros(embedding.embedding_length))
            continue

        n_w = 0
        try:
            sentence = Sentence(text)
            embedding.embed(sentence)
        except:
            try:
                sentence = Sentence(" ".join(text.split()[:512]))
                embedding.embed(sentence)
            except:
                result.append(np.zeros(embedding.embedding_length))
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
    @contextmanager
    def timer(name):
        t0 = time.time()
        yield


    with timer('bert'):
        embedding = BertEmbeddings('bert-base-uncased')
        result = w2v_flair(train["Description"], embedding, name="bert")
        result.to_feather("../feature/bert_flair.feather")