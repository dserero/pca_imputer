import numpy as np
import pandas as pd

from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class PCAImputer(_BaseImputer):
    def __init__(self, *, n_components):
        self.n_components = n_components
    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return self._fill_db(X, self.n_components)
    
    @classmethod
    def _fill_db(cls, db, n_components):
        db_ = db.copy()
        threshs = sorted(list(set((~db_.isnull()).sum())))[::-1]
        threash_0, thread_1 = threshs[0], threshs[1]
        X = db_.dropna(thresh = threash_0, axis = 1).dropna()
        y = db_.dropna(thresh = thread_1, axis = 1).dropna()
        y_cols = list(set(y.columns) - set(X.columns))
        y = y[y_cols]
        y_new = cls._fill_y(X, y, n_components)
        db_[y_cols] = db_[y_cols].mask(db_[y_cols].isnull(), y_new, axis=0)
        if db_.isnull().sum().sum() > 0:
            return cls._fill_db(db_, n_components)
        return db_

    @classmethod
    def _fill_y(cls, X, y, n_components):
        missing_y_index = X.index.difference(y.index)
        idx_in_common = X.dropna().index.intersection(y.dropna().index)
        X_train, y_train = X.loc[idx_in_common], y.loc[idx_in_common]
        reg = Pipeline(steps=[
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=min(n_components, X_train.shape[1]))),
                    ('reg', LinearRegression())
        ])
        reg.fit(X_train, y_train)
        pred_y = pd.DataFrame(reg.predict(X.loc[missing_y_index]), index = missing_y_index, columns=y_train.columns)
        if pred_y.isnull().sum().sum() > 0:
            raise ValueError("Pred y has missing values")
        y_new = pd.concat([y, pred_y]).sort_index()
        # if y_new.isnull().sum().sum() > 0:
        #     raise ValueError("y_new has missing values")
        assert len(y_new) == len(X)
        return y_new