# coding: UTF-8
# adapted from https://github.com/arosh/BM25Transformer/blob/master/bm25.py

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency
from utils import *

class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X


with timer('merge additional files'):
    train = merge_breed_name(train)

orig_cols = train.columns
train[text_features].fillna('missing', inplace=True)
with timer('tfidf_bm25 + svd / nmf'):
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
        TruncatedSVD(n_components=n_components, random_state=seed)
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['tfidf_bm25_svd_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer;
    gc.collect()

with timer('count_bm25 + svd / nmf'):
    vectorizer = make_pipeline(
        CountVectorizer(),
        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
        TruncatedSVD(n_components=n_components, random_state=seed),
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['count_bm25_svd_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer;
    gc.collect()

with timer('tfidf2_bm25 + svd / nmf'):
    vectorizer = make_pipeline(
        TfidfVectorizer(min_df=2, max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english'),
        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
        TruncatedSVD(n_components=n_components, random_state=seed)
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['tfidf2_bm25_svd_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer;
    gc.collect()

with timer('count2_bm25 + svd / nmf'):
    vectorizer = make_pipeline(
        CountVectorizer(min_df=2, max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), stop_words='english'),
        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
        TruncatedSVD(n_components=n_components, random_state=seed)
    )
    X = vectorizer.fit_transform(train['Description'])
    X = pd.DataFrame(X, columns=['count2_bm25_svd_{}'.format(i) for i in range(n_components)])
    train = pd.concat([train, X], axis=1)
    del vectorizer;
    gc.collect()
new_cols = [c for c in train.columns if c not in orig_cols]
train[new_cols].to_feather("../feature/bm25.feather")