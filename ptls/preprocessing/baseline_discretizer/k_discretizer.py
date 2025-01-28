from sklearn.preprocessing import KBinsDiscretizer

class KDiscretizer():
    def __init__(self, f_names, k_bins, d_type='quantile'):
        '''
        d_type should be one of {'quantile', 'ordinal', ‘kmeans’}
        '''
        self.f_names = f_names
        self.disc = KBinsDiscretizer(
            n_bins=k_bins, encode='ordinal', strategy=d_type)

    def fit_transform(self, X):
        X.loc[:, self.f_names] = self.disc.fit_transform(X.loc[:, self.f_names]).astype(int)
        return X

    def transform(self, X):
        X.loc[:, self.f_names] = self.disc.transform(X.loc[:, self.f_names]).astype(int)
        return X