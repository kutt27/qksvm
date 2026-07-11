import numpy as np


class MinMaxScaler:
    """
    Scales data feature-wise to a range, default [-pi, pi].

    This is needed because quantum gates operate on angles (radians),
    and the feature map's Rz/Ry gates expect values in a reasonable
    angular range.
    """

    def __init__(self, low=-np.pi, high=np.pi):
        self.low = low
        self.high = high
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        # Avoid division by zero for constant features
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.data_min_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call .fit() first.")
        # Min-max to [0, 1], then scale to [low, high]
        X_scaled = (X - self.data_min_) / self.data_range_
        return X_scaled * (self.high - self.low) + self.low

    def fit_transform(self, X):
        return self.fit(X).transform(X)
