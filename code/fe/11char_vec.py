from utils import *

with timer('tfidf3 + svd / nmf'):
    vectorizer = make_pipeline(
        TfidfVectorizer(min_df=30, max_features=50000, binary=True,
                        strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                        ngram_range=(3, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            n_jobs=2,
        ),
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['tfidf3_svd_{}'.format(i) for i in range(n_components)]
                                + ['tfidf3_nmf_{}'.format(i) for i in range(n_components)])
    X.to_feather("../feature/tfidf3.feather")

with timer('count3 + svd / nmf'):
    vectorizer = make_pipeline(
        CountVectorizer(min_df=30, max_features=50000, binary=True,
                        strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                        ngram_range=(3, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=n_components, random_state=seed),
            NMF(n_components=n_components, random_state=seed),
            n_jobs=1,
        ),
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['count3_svd_{}'.format(i) for i in range(n_components)]
                                + ['count3_nmf_{}'.format(i) for i in range(n_components)])
    X.to_feather("../feature/count3.feather")