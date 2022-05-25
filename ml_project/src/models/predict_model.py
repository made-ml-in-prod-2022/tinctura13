from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SklearnClassifierModel = Union[LogisticRegression, RandomForestClassifier]


def predict_model(
    model: SklearnClassifierModel,
    features: pd.DataFrame
) -> np.ndarray:
    """Makes predictions and probabilities"""
    predictions = model.predict(features)
    probas = model.predict_proba(features)
    return predictions, probas
