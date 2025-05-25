from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

from ..features.stats import *


class ACCStatsTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn Transformer to computes the statistics for each ACC sample
    """
    def __init__(self, single_ax_stat_list="all", multi_ax_stat_list="all"):
        self.single_ax_stat_list = single_ax_stat_list
        self.multi_ax_stat_list = multi_ax_stat_list

    def fit(self, data, *args, **kwargs):
        return self

    def transform(self, data):
        X = data[:, 0::3]
        Y = data[:, 1::3]
        Z = data[:, 2::3]
        norm = (X ** 2 + Y ** 2 + Z ** 2) ** .5

        features = {}

        for f_name, f in single_axis_features.items():
            if self.single_ax_stat_list == "all" or f_name in self.single_ax_stat_list:
                features[f"{f_name}_x"] = f(X)
                features[f"{f_name}_y"] = f(Y)
                features[f"{f_name}_z"] = f(Z)
                features[f"{f_name}_norm"] = f(norm)

        for f_name, f in multiple_axis_features.items():
            if self.multi_ax_stat_list == "all" or f_name in self.multi_ax_stat_list:
                features[f_name] = f(X, Y, Z, norm)

        return pd.DataFrame(features)


