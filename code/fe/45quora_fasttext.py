from utils import *

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


with timer('quora fasttext meta'):
    embedding = "../../input/quora-embedding/GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(embedding, binary=True)
    X = pretrained_w2v(train["Description_Emb"], model, name="quora_fasttext")
    X.to_feather("../feature/quora_fasttext.feather")

with timer('quora fasttext meta'):
    train["desc_emb"] = [analyzer_embed(text) for text in train["desc"]]
    X_meta = pretrained_w2v(train["desc_emb"], model, name="quora_fasttext_meta")
    X_meta.to_feather("../feature/quora_fasttext_meta.feather")

