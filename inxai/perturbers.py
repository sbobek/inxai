from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np



class NormalNoisePerturber(TransformerMixin, BaseEstimator):
    """ Generates normal noise according to specified parameters.
    During perturbation phase the noise is added to the features.
    If @scale is not given, it is learned during @fit.
    """
    def __init__(self, scale=None, importances = None):
        self.scale = scale
        self.importances = importances
        self.colnames = None

    def set_importances(self, importances):
        self.importances = importances

    def fit(self, X):
        if self.scale is None:
            #detect dscale for every column
            pass
        self.colnames = X.columns
        return self

    def transform(self, X):
        return X.apply(lambda x: x + self.importances * np.random.normal(0, self.scale), axis=1)

    def get_feature_names(self):
        return self.colnames

class MultivarietNormalNoisePerturber(TransformerMixin, BaseEstimator):
    def __init__(self,  mean=None, cov=None, importances = None):
        self.mean = mean
        self.cov = cov
        self.importances = importances
        self.colnames = None

    def set_importances(self, importances):
        self.importances = importances

    def fit(self, X):
        if self.mean or self.cov is None:
            #detect dscale for every column
            pass
        self.colnames = X.columns
        return self

    def transform(self, X):
        #TODO normal dist in lc of x and scale of importance should be multiplied by the multiivariate dist
        return X.apply(lambda x: x + self.importances * np.random.multivariate_normal(self.mean, self.cov), axis=1)

    def get_feature_names(self):
        return self.colnames


class CategoricalNoisePerturber(TransformerMixin, BaseEstimator):
    def __init__(self, probability_multiplier=1, importances = None):
        self.probability_multiplier = probability_multiplier
        self.importances = importances

    def set_importances(self, importances):
        self.importances = importances

    def fit(self, X):
        return self

    def transform(self, X):
        for col_idx, column in enumerate(X):
            unique_elements = X[column].unique()
            for row in X[column].iteritems():
                if np.random.random() < self.probability_multiplier * self.importances[col_idx]:
                    X.loc[row[0], column] = np.random.choice(unique_elements)
        return X


class ShufflePerturber(TransformerMixin, BaseEstimator):
    """Perturbation that is performed according to the distribution of the features by shuffling values within features
    across all instances. No artificial noise is generated
    """
    def fit(self, X):
        return self

    def transform(self, X):
        pass
