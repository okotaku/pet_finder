from utils import *
os.system('python ../input/jieba-package/repository/fxsjy-jieba-8212b6c/setup.py install')
import jieba
import sys
from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), ' ')
kanji = re.compile('[一-龥]')

def analyzer_bow(text):
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from',
                  'in', 'on']
    text = text.lower()  # 小文字化
    text = text.replace('\n', '')  # 改行削除
    text = text.replace('\t', '')  # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    for bad_word in contraction_mapping:
        if bad_word in text:
            text = text.replace(bad_word, contraction_mapping[bad_word])
    if re.search(kanji, text):
        text = " ".join([word for word in jieba.cut(text) if word != ' '])
    text = text.split(' ')  # スペースで区切る
    text = [sb.stem(t) for t in text]

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):  # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if word in stop_words:  # ストップワードに含まれるものは除外
            continue
        if len(word) < 2:  # 1文字、0文字（空文字）は除外
            continue
        words.append(word)

    return " ".join(words)


def analyzer_embed(text):
    text = text.lower()  # 小文字化
    text = text.replace('\n', '')  # 改行削除
    text = text.replace('\t', '')  # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    for bad_word in contraction_mapping:
        if bad_word in text:
            text = text.replace(bad_word, contraction_mapping[bad_word])
    if re.search(kanji, text):
        text = " ".join([word for word in jieba.cut(text) if word != ' '])
    text = text.split(' ')  # スペースで区切る

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):  # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if len(word) < 1:  # 0文字（空文字）は除外
            continue
        words.append(word)

    return " ".join(words)

train["Description_Emb2"] = [analyzer_embed(text) for text in train["Description"]]
train["Description_bow2"] = [analyzer_bow(text) for text in train["Description"]]

train = train.reset_index(drop=True)
orig_cols = train.columns
with timer('tfidf + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['tfidf_svd_{}'.format(i) for i in range(n_components)]
                     + ['tfidf_nmf_{}'.format(i) for i in range(n_components)]
                    + ['tfidf_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('count + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        CountVectorizer(),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['count_svd_{}'.format(i) for i in range(n_components)]
                     + ['count_nmf_{}'.format(i) for i in range(n_components)]
                    + ['count_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('tfidf2 + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        TfidfVectorizer(min_df=2,  max_features=20000,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['tfidf2_svd_{}'.format(i) for i in range(n_components)]
                     + ['tfidf2_nmf_{}'.format(i) for i in range(n_components)]
                    + ['tfidf2_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('count2 + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        CountVectorizer(min_df=2,  max_features=20000,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['count2_svd_{}'.format(i) for i in range(n_components)]
                     + ['count2_nmf_{}'.format(i) for i in range(n_components)]
                    + ['count2_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('tfidf3 + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        TfidfVectorizer(min_df=30, max_features=50000, binary=True,
                        strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                        ngram_range=(3, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['tfidf3_svd_{}'.format(i) for i in range(n_components)]
                                + ['tfidf3_nmf_{}'.format(i) for i in range(n_components)]
                    + ['tfidf3_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('count3 + svd / nmf / bm25'):
    vectorizer = make_pipeline(
        CountVectorizer(min_df=30, max_features=50000, binary=True,
                        strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                        ngram_range=(3, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=n_components, random_state=seed)
            ),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description_bow2'])
    X = pd.DataFrame(X, columns=['count3_svd_{}'.format(i) for i in range(n_components)]
                                + ['count3_nmf_{}'.format(i) for i in range(n_components)]
                    + ['count3_bm25_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer; gc.collect()

with timer('description fasttext'):
    embedding = '../../input/quora-embedding/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(embedding, binary=True)
    X = pretrained_w2v(train["Description_Emb2"], model, name="gnvec")
    train = pd.concat([train, X], axis=1)
    del model; gc.collect()

with timer('description glove'):
    embedding = "../../input/pymagnitude-data/glove.840B.300d.magnitude"
    model = Magnitude(embedding)
    X = w2v_pymagnitude(train["Description_Emb2"], model, name="glove")
    train = pd.concat([train, X], axis=1)
    del model; gc.collect()
new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("new_text_feats.feather")