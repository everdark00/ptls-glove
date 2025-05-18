import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimePreprocessor:
    def __init__(self, idcol, ordercol, datecol=None, timecol=None, mode='all', exclude_list=[], scale_numeric=False):
        """
        ordercol: must contain positive integers
        datecol: must contain transaction date in format of string YYYY-MM-DD
        timecol: must contain transaction time in format of string HH:MM:SS
        mode: 'all' - collect all features, 'cat' - collect categorical features, 'num' - collect numeric features
        """
        self.idcol = idcol
        self.ordercol = ordercol
        self.datecol = datecol
        self.timecol = timecol
        self.mode = mode
        self.scale_numeric=scale_numeric

        self.cat_features = []
        self.num_features = []
        self.exclude_list = exclude_list

        self.nonperiodic = []

        self.scalers = dict()

    def time_to_seconds(self, t):
        h, m, s = map(int, t.split(':'))
        return h * 3600 + m * 60 + s

    def normalize(self, f):
        return  (X[self.ordercol] - X[self.ordercol].mean())/(X[self.ordercol].max() - X[self.ordercol].min())

    def fit(self, X):
        X = X[[self.idcol, self.ordercol] + ([self.datecol] if self.datecol is not None else []) +  ([self.timecol] if self.timecol is not None else [])]
        if self.ordercol is not None:
            ordercol_deltas = X.groupby(self.idcol)[self.ordercol].transform(lambda x: x.sort_values().diff()).fillna(1)
            first_values = X.groupby(self.idcol)[self.ordercol].agg('min')
            if (ordercol_deltas != 1.0).sum() != 0 or  (first_values == first_values.iloc[0]).sum() != 0:
                if X[self.ordercol].unique().shape[0] < 1000:
                    self.cat_features.append('TIME_ordercol_cat')
                    self.nonperiodic.append('TIME_ordercol_cat')
                else:
                    self.num_features.append('TIME_ordercol_num')
                    if self.scale_numeric:
                        scaler = MinMaxScaler()
                        scaler.fit(X[self.ordercol].values.reshape(-1, 1))
                        self.scalers['TIME_ordercol_num'] = scaler
                    self.nonperiodic.append('TIME_ordercol_num')

        if self.timecol is not None:
            if self.datecol is not None:
                self.num_features.append('TIME_unix_seconds')
                self.nonperiodic.append('IME_unix_seconds')

            self.num_features.append('TIME_daily_seconds')

            if self.scale_numeric:
                time_seconds = X[self.timecol].apply(self.time_to_seconds)
                scaler = MinMaxScaler()
                scaler.fit(time_seconds.values.reshape(-1, 1))
                self.scalers['TIME_daily_seconds'] = scaler

        if self.datecol is not None:
            dates = pd.to_datetime(X[self.datecol], format='%Y-%m-%d')
            date_range = (max(dates) - min(dates)).days
            if date_range > 366:
                self.cat_features.append('TIME_month')
            if date_range > 90:  
                self.cat_features.append('TIME_monthday')
            if date_range > 30:
                self.cat_features.append('TIME_weekday')

        self.cat_features = sorted(list(set(self.cat_features) - set(self.exclude_list)))
        self.num_features = sorted(list(set(self.num_features) - set(self.exclude_list)))


    def transform(self, X):
        if self.datecol is not None:
            if self.timecol is not None:
                timestamp = pd.to_datetime(X[self.datecol] + " " + X[self.timecol], format='%Y-%m-%d %H:%M:%S')
            else:
                timestamp = pd.to_datetime(X[self.datecol], format='%Y-%m-%d')
        elif self.timecol is not None:
            timestamp = pd.to_datetime(X[self.timecol], format='%H:%M:%S')

        if self.mode == 'cat' or self.mode == 'all':
            if 'TIME_ordercol_cat' in self.cat_features:
                X['TIME_ordercol_cat'] = X[self.ordercol] 

            if 'TIME_month' in self.cat_features:
                X['TIME_month'] =  timestamp.dt.month

            if 'TIME_monthday' in self.cat_features:
                X['TIME_monthday'] =  timestamp.dt.day

            if 'TIME_weekday' in self.cat_features:
                X['TIME_weekday'] =  timestamp.dt.weekday
            
        if self.mode == 'num' or self.mode == 'all':
            if 'TIME_ordercol_num' in self.num_features:
                if self.scale_numeric:
                    X['TIME_ordercol_num'] =  self.scalers['TIME_ordercol_num'].transform(X[self.ordercol].values.reshape(-1, 1))
                else:
                    X['TIME_ordercol_num'] = X[self.ordercol]

            if 'TIME_daily_seconds' in self.num_features:
                time_seconds = X[self.timecol].apply(self.time_to_seconds)
                if self.scale_numeric:
                    X['TIME_daily_seconds'] =  self.scalers['TIME_daily_seconds'].transform(time_seconds.values.reshape(-1, 1))
                else:
                    X['TIME_daily_seconds'] = time_seconds

            if 'TIME_unix_seconds' in self.num_features:
                timestamp_sec = timestamp.astype('int64') // 1e+9

                X['TIME_unix_seconds'] = timestamp_sec

        if self.datecol is not None:
            X = X.drop(columns=[self.datecol])
        if self.timecol is not None:
            X = X.drop(columns=[self.timecol])

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



# class TimePreprocessor:
#     def __init__(self, idcol, ordercol, datecol=None, timecol=None, mode='all', exclude_list=[]):
#         """
#         ordercol: must contain positive integers
#         datecol: must contain transaction date in format of string YYYY-MM-DD
#         timecol: must contain transaction time in format of string HH:MM:SS
#         mode: 'all' - collect all features, 'cat' - collect categorical features, 'num' - collect numeric features
#         """
#         self.idcol = idcol
#         self.ordercol = ordercol
#         self.datecol = datecol
#         self.timecol = timecol
#         self.mode = mode

#         self.cat_features = []
#         self.num_features = []
#         self.exclude_list = exclude_list

#         self.scalers = dict()

#     def time_to_seconds(self, t):
#         h, m, s = map(int, t.split(':'))
#         return h * 3600 + m * 60 + s

#     def normalize(self, f):
#         return  (X[self.ordercol] - X[self.ordercol].mean())/(X[self.ordercol].max() - X[self.ordercol].min())

#     def fit(self, X):
#         X = X[[self.idcol, self.ordercol, self.datecol, self.timecol]]
#         if self.ordercol is not None:
#             ordercol_deltas = X.groupby(self.idcol)[self.ordercol].transform(lambda x: x.sort_values().diff()).fillna(1)
#             first_values = X.groupby(self.idcol)[self.ordercol].agg('min')
#             if (ordercol_deltas != 1.0).sum() != 0 or  (first_values == first_values.iloc[0]).sum() != 0:
#                 self.num_features.append('TIME_ordercol_num')
#                 scaler = MinMaxScaler()
#                 scaler.fit(X[self.ordercol].values.reshape(-1, 1))
#                 self.scalers['TIME_ordercol_num'] = scaler
#                 if X[self.ordercol].unique().shape[0] < 1000:
#                     self.cat_features.append('TIME_ordercol_cat')

#         if self.timecol is not None:
#             X['time_seconds'] = X[self.timecol].apply(self.time_to_seconds)
#             if self.datecol is not None:
#                 time_deltas = X.sort_values(by=[self.idcol, self.datecol, 'time_seconds']).groupby([self.idcol, self.datecol])['time_seconds'].diff().fillna(3601)
#                 if (time_deltas < 3600).sum() > 0.1 * time_deltas.shape[0]:
#                     self.num_features.append('TIME_daily_seconds_sin')
#                     self.num_features.append('TIME_daily_seconds_cos')
#             else:
#                 self.num_features.append('TIME_time_seconds')
                    
#             self.cat_features.append('TIME_hour')
#             self.num_features.append('TIME_hour_sin')
#             self.num_features.append('TIME_hour_cos')
#         if self.datecol is not None:
#             dates = pd.to_datetime(X[self.datecol], format='%Y-%m-%d')
#             date_range = (max(dates) - min(dates)).days
#             if date_range > 366:
#                 self.cat_features.append('TIME_month')
#             if date_range > 90:  
#                 self.cat_features.append('TIME_monthday')
#                 self.num_features.append('TIME_monthday_sin')
#                 self.num_features.append('TIME_monthday_cos')
#             if date_range > 30:
#                 self.cat_features.append('TIME_weekday')
#                 self.num_features.append('TIME_weekday_sin')
#                 self.num_features.append('TIME_weekday_cos')
#         self.cat_features = sorted(list(set(self.cat_features) - set(self.exclude_list)))
#         self.num_features = sorted(list(set(self.num_features) - set(self.exclude_list)))


#     def transform(self, X):
#         if self.datecol is not None:
#             if self.timecol is not None:
#                 timestamp = pd.to_datetime(X[self.datecol] + " " + X[self.timecol], format='%Y-%m-%d %H:%M:%S')
#             else:
#                 timestamp = pd.to_datetime(X[self.datecol], format='%Y-%m-%d')
#         elif self.timecol is not None:
#             timestamp = pd.to_datetime(X[self.timecol], format='%H:%M:%S')

#         if len({'TIME_daily_seconds_sin', 'TIME_daily_seconds_cos'} & set(self.num_features)) > 0:
#             time_seconds = X[self.timecol].apply(self.time_to_seconds)

#         if self.mode == 'cat' or self.mode == 'all':
#             if 'TIME_ordercol_cat' in self.cat_features:
#                 X['TIME_ordercol_cat'] = X[self.ordercol] 

#             if 'TIME_hour' in self.cat_features:
#                 X['TIME_hour'] =  timestamp.dt.hour

#             if 'TIME_month' in self.cat_features:
#                 X['TIME_month'] =  timestamp.dt.month

#             if 'TIME_monthday' in self.cat_features:
#                 X['TIME_monthday'] =  timestamp.dt.day

#             if 'TIME_weekday' in self.cat_features:
#                 X['TIME_weekday'] =  timestamp.dt.weekday
            
#         if self.mode == 'num' or self.mode == 'all':
#             if 'TIME_ordercol_num' in self.num_features:
#                 X['TIME_ordercol_num'] =  self.scalers['TIME_ordercol_num'].transform(X[self.ordercol].values.reshape(-1, 1))

#             if 'TIME_daily_seconds_sin' in self.num_features:
#                 X['TIME_daily_seconds_sin'] =  np.sin(2 * np.pi * time_seconds / (60 * 60 * 24))

#             if 'TIME_daily_seconds_cos' in self.num_features:
#                 X['TIME_daily_seconds_cos'] =  np.cos(2 * np.pi * time_seconds / (60 * 60 * 24))

#             if 'TIME_hour_sin' in self.num_features:
#                 X['TIME_hour_sin'] =  np.sin(2 * np.pi * timestamp.dt.hour.values / 24)

#             if 'TIME_hour_cos' in self.num_features:
#                 X['TIME_hour_cos'] =  np.cos(2 * np.pi * timestamp.dt.hour.values / 24)

#             if 'TIME_monthday_sin' in self.num_features:
#                 X['TIME_monthday_sin'] =  np.sin(2 * np.pi * timestamp.dt.day.values / 30)

#             if 'TIME_monthday_cos' in self.num_features:
#                 X['TIME_monthday_cos'] =  np.cos(2 * np.pi * timestamp.dt.day.values / 30)

#             if 'TIME_weekday_sin' in self.num_features:
#                 X['TIME_weekday_sin'] =  np.sin(2 * np.pi * timestamp.dt.weekday.values / 7)

#             if 'TIME_weekday_cos' in self.num_features:
#                 X['TIME_weekday_cos'] =  np.cos(2 * np.pi * timestamp.dt.weekday.values / 7)

#         return X

#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)