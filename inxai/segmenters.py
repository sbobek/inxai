from sklearn.base import TransformerMixin, BaseEstimator


class BaseSegmenter(TransformerMixin, BaseEstimator):

    def fit(self, X):
        return self

    def transform(self, X):
        return X



