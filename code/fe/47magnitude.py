from utils import *


def w2v_pymagnitude(train_text, model, name):
    train_corpus = [text_to_word_sequence(text) for text in train_text]

    result = []
    for text in train_corpus:
        vec = np.zeros(model.dim)
        for n_w, word in enumerate(text):
            if word in model:  # 0.9906
                vec = vec + model.query(word)
                continue
            word_ = word.upper()
            if word_ in model:  # 0.9909
                vec = vec + model.query(word_)
                continue
            word_ = word.capitalize()
            if word_ in model:  # 0.9925
                vec = vec + model.query(word_)
                continue
            word_ = ps.stem(word)
            if word_ in model:  # 0.9927
                vec = vec + model.query(word_)
                continue
            word_ = lc.stem(word)
            if word_ in model:  # 0.9932
                vec = vec + model.query(word_)
                continue
            word_ = sb.stem(word)
            if word_ in model:  # 0.9933
                vec = vec + model.query(word_)
                continue
            vec = vec + model.query(word)

        vec = vec / (n_w + 1)
        result.append(vec)

    w2v_cols = ["{}_{}".format(name, i) for i in range(1, model.dim + 1)]
    result = pd.DataFrame(result)
    result.columns = w2v_cols
    del model;
    gc.collect()

    return result

with timer('quora fasttext meta'):
    embedding = "/mnt/embedding/GoogleNews-vectors-negative300_light.magnitude"
    model = Magnitude(embedding)
    X = w2v_pymagnitude(train["Description_Emb"], model, name="googlenews_mag_light")
    X.to_feather("../feature/googlenews_mag_light.feather")

with timer('quora fasttext meta'):
    train["desc_emb"] = [analyzer_embed(text) for text in train["desc"]]
    X_meta = w2v_pymagnitude(train["desc_emb"], model, name="googlenews_mag_ligh_tmeta")
    X_meta.to_feather("../feature/googlenews_mag_ligh_tmeta.feather")

with timer('quora fasttext meta'):
    embedding = "/mnt/embedding/GoogleNews-vectors-negative300_medium.magnitude"
    model = Magnitude(embedding)
    X = w2v_pymagnitude(train["Description_Emb"], model, name="googlenews_mag_medium")
    X.to_feather("../feature/googlenews_mag_medium.feather")

with timer('quora fasttext meta'):
    #train["desc_emb"] = [analyzer_embed(text) for text in train["desc"]]
    X_meta = w2v_pymagnitude(train["desc_emb"], model, name="googlenews_mag_medium_meta")
    X_meta.to_feather("../feature/googlenews_mag_medium_meta.feather")

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
    columns = [c for c in sentiment_features.columns if c != 'sentiment_text']
    g = meta_parser.sentiment_files[list(sentiment_features.columns) + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

    columns = [c for c in meta_features.columns if c != 'annots_top_desc']
    g = meta_parser.metadata_files[columns + ['PetID']].groupby('PetID').agg(stats)
    g.columns = [c + '_' + stat for c in columns for stat in stats]
    train = train.merge(g, how='left', on='PetID')

with timer('metadata, annots_top_desc'):
    meta_features = meta_parser.metadata_files[['PetID', 'annots_top_desc']]
    meta_features = meta_features.groupby('PetID')['annots_top_desc'].sum().reset_index()
    train = train.merge(meta_features, how='left', on='PetID')

    sentiment_features = meta_parser.sentiment_files[['PetID', 'sentiment_text']]
    sentiment_features = sentiment_features.groupby('PetID')['sentiment_text'].sum().reset_index()
    train = train.merge(sentiment_features, how='left', on='PetID')

    train['desc'] = ''
    for c in ['BreedName_main_breed', 'BreedName_second_breed', 'annots_top_desc', 'sentiment_text']:
        train['desc'] += ' ' + train[c].astype(str)

