from utils import *


def pretrained_w2v(train_text, embedding, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    model = KeyedVectors.load_word2vec_format(embedding, binary=True)

    result = []
    for text in train_corpus:
        n_skip = 0
        vec = np.zeros(model.vector_size)
        for n_w, word in enumerate(text):
            try:
                vec_ = model.wv[word]
            except:
                n_skip += 1
                continue
            if n_w - n_skip == 0:
                vec = vec_
            else:
                vec = vec + vec_
        vec = vec / (n_w - n_skip + 1)
        result.append(vec)

    w2v_cols = ["{}{}".format(name, i) for i in range(1, model.vector_size + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols
    del model;
    gc.collect()

    return result


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
    del model;
    gc.collect()

    return result


with timer('merge additional files'):
    train = merge_breed_name(train)

with timer('metadata'):
    # TODO: parallelization
    meta_parser = MetaDataParser()
    sentiment_features = meta_parser.sentiment_files['sentiment_filename'].apply(
        lambda x: meta_parser._transform(x, sentiment=True))
    meta_parser.sentiment_files = pd.concat([meta_parser.sentiment_files, sentiment_features], axis=1, sort=False)
    meta_features = meta_parser.metadata_files['metadata_filename'].apply(
        lambda x: meta_parser._transform(x, sentiment=False))
    meta_parser.metadata_files = pd.concat([meta_parser.metadata_files, meta_features], axis=1, sort=False)

    stats = ['mean', 'min', 'max', 'median']
    g = meta_parser.sentiment_files[list(sentiment_features.columns) + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in sentiment_features.columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

    columns = [c for c in meta_features.columns if c != 'annots_top_desc']
    g = meta_parser.metadata_files[columns + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

with timer('metadata, annots_top_desc'):
    meta_features = meta_parser.metadata_files[['PetID', 'annots_top_desc']]
    meta_features = meta_features.groupby('PetID')['annots_top_desc'].sum().reset_index()
    train = train.merge(meta_features, how='left', on='PetID')

    train['desc'] = ''
    for c in ['BreedName_main_breed', 'BreedName_second_breed', 'annots_top_desc']:
        train['desc'] += ' ' + train[c].astype(str)

with timer('fasttext meta'):
    embedding = '../../input/quora-embedding/GoogleNews-vectors-negative300.bin'
    X = pretrained_w2v(train["desc"], embedding, name="fast_meta")
    X.to_feather("../feature/fast_meta.feather")

with timer('glove meta'):
    embedding = "../../input/pymagnitude-data/glove.840B.300d.magnitude"
    X = w2v_pymagnitude(train["desc"], embedding, name="glove_meta")
    X.to_feather("../feature/glove_meta.feather")