import pickle
from typing import Dict, NoReturn, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SklearnClassifierModel = Union[LogisticRegression, RandomForestClassifier]


def evaluate_model(
    predictions: np.ndarray,
    probas: np.ndarray,
    target: pd.Series
) -> Dict[str, float]:
    """Calculate model's metrics"""
    return {
        "accuracy_score": accuracy_score(target, predictions),
        "f1_score": f1_score(target, predictions),
        "roc_auc_score": roc_auc_score(target, probas[:, 1]),
    }


def serialize_model(
    model: SklearnClassifierModel,
    output: str
) -> NoReturn:
    """Save model to storage"""
    with open(output, "wb") as fout:
        pickle.dump(model, fout)


def deserialize_model(
    input_model: str
) -> SklearnClassifierModel:
    """Load model from storage"""
    with open(input_model, "rb") as fin:
        model = pickle.load(fin)
    return model
