from sklearn.tree import DecisionTreeClassifier

class SingleTreeDiscretizer():
    def __init__(self, f_names, target_name, target_type, k_bins=None, pruning_rates=None, d_type='quantile'):
        '''
        d_type should be one of {'quantile', 'ordinal', ‘kmeans’}
        '''
        self.target_name = target_name
        self.target.type = target_type
        self.pruning_rates = pruning_rates
        self.f_names = f_names
        self.k_bins = k_bins
        self.tresholds = {fn : [] for fn in self.f_names}

    def fit_transform(self, X):
        for i, fn in enumerate(self.f_names): 
            if self.k_bins is not None:
                ccp_alpha_min = 1e-7
                ccp_alpha_max = 1e-3

                k_bins_f = self.k_bins[i]
                curr_ccp_alpha = (ccp_alpha_max + ccp_alpha_min) / 2.0
                tresholds = None
                for i in range(10):
                    if self.target_type == 'classification':
                        disc = DecisionTreeClassifier(
                            criterion='gini',
                            ccp_alpha=curr_ccp_alpha,
                        )
                    else:
                        disc = DecisionTreeRegressor(
                            criterion='gini',
                            ccp_alpha=curr_ccp_alpha,
                        )
                    disc.fit(X=X.loc[:, [fn]], y=X[self.target_name])
                    curr_k_bins = disc.tree_.threshold[disc.tree_.threshold != -2].shape[0] + 1
                    if curr_k_bins < k_bins_f:
                        ccp_alpha_max = curr_ccp_alpha
                    elif curr_k_bins > k_bins_f + 0.05 * k_bins:
                        ccp_alpha_min = curr_ccp_alpha
                    else:
                        tresholds = sorted(disc.tree_.threshold[disc.tree_.threshold != -2])
                        break
                
                bins_gap = pd.DataFrame({'bn' : tresholds[:-1], 
                                         'gap' : [tresholds[i] - tresholds[i - 1] for i in range(1, tresholds.shape[0])]})
                bins_gap = bins_gap.sort_values(by='gap', ascending=True)

                for j in range(tresholds.shape[0] + 1 - k_bins_f):
                    bins_gap.iloc[j, :] = np.nan
                bins_gap = bins_gap.dropna()
                self.tresholds[fn] = list(sorted(bins_gap.bn.values)) + [tresholds[-1]]


            else:
                if self.pruning_rates is not None:
                    ccp_alpha = self.pruning_rates[i]
                else:
                    ccp_alpha = 1e-4
                
                if self.target_type == 'classification':
                        disc = DecisionTreeClassifier(
                            criterion='gini',
                            ccp_alpha=ccp_alpha,
                        )
                else:
                    disc = DecisionTreeRegressor(
                        criterion='gini',
                        ccp_alpha=ccp_alpha,
                    )
                disc.fit(X=X.loc[:, [fn]], y=X[self.target_name])
                self.tresholds[fn] = list(sorted(disc.tree_.threshold[disc.tree_.threshold != -2]))

        return self.transform(X)

    def transform(self, X):
        for i, fn in enumerate(self.f_names): 
            X.loc[(X[fn] <= self.tresholds[fn][0]), fn] = 0
            for j in range(1, len(self.tresholds[fn])):
                X.loc[(X[fn] > self.tresholds[fn][j - 1]) * (X[fn] <= self.tresholds[fn][j]), fn] = j
            X.loc[(X[fn] > self.tresholds[fn][-1]), fn] = len(self.tresholds[fn])
            X[fn] = data[fn].astype("int64")
    
        return X

