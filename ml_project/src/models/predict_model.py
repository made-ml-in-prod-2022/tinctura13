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
    predictions = model.predict(features)
    return predictions
