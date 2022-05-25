import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TransformException(Exception):
    pass


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, x: np.ndarray) -> "CustomTransformer":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise TransformException("make fit before transfrom")
        else:
            x = (x - self.mean) / self.std
            return x
