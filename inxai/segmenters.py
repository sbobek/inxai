from sklearn.base import TransformerMixin, BaseEstimator


class BaseSegmenter(TransformerMixin, BaseEstimator):

    def fit(self, X):
        return self

    def transform(self, X):
        Xs= X.apply(lambda x: BaseSegment(x))
        return Xs



class BaseSegment:
    vlaue=None
    def __init__(self, value):
        self.vlaue=value