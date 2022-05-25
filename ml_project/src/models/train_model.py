from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.configs.train_params import LRParams, RFParams

SklearnClassifierModel = Union[LogisticRegression, RandomForestClassifier]


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: Union[LRParams, RFParams]
) -> SklearnClassifierModel:
    if params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty=params.penalty,
            tol=params.tol,
            C=params.C,
            random_state=params.random_state,
        )
    elif params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            criterion=params.criterion,
            random_state=params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model
