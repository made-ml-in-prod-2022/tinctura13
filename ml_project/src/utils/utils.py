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
    target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predictions),
        "f1_score": f1_score(target, predictions),
        "roc_auc_score": roc_auc_score(target, predictions),
    }


def serialize_model(
    model: SklearnClassifierModel,
    output: str
) -> NoReturn:
    with open(output, "wb") as fout:
        pickle.dump(model, fout)


def deserialize_model(
    input_model: str
) -> SklearnClassifierModel:
    with open(input_model, "rb") as fin:
        model = pickle.load(fin)
    return model
