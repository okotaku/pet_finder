from utils import *


def w2v(train_text, w2v_params, embed_path):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = KeyedVectors.load_word2vec_format(embed_path, binary=True)

    model2 = word2vec.Word2Vec(**w2v_params)
    model2.build_vocab(train_corpus)
    total_examples = model2.corpus_count
    model2.build_vocab([list(model.vocab.keys())], update=True)
    model2.intersect_word2vec_format(embed_path, binary=True, lockf=1.0)
    model2.train(train_corpus, total_examples=total_examples, epochs=model2.iter)
    model2.intersect_word2vec_format(embed_path, binary=True, lockf=1.0)

    result = []
    for text in train_corpus:
        n_skip = 0
        for n_w, word in enumerate(text):
            try:
                vec_ = model2.wv[word]
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
    embed_path = '../../input/quora-embedding/GoogleNews-vectors-negative300.bin'
    result = w2v(train["Description_Emb"], w2v_params, embed_path)
    result.to_feather("../feature/gnvec_finetune_v2.feather")