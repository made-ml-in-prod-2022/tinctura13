import logging
import os
import pickle
from typing import List, Union

import pandas as pd
import yaml
from pydantic import BaseModel, conlist
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.features import CAT_FEATS, COL_ORDER, NUM_FEATS

SkleratnClassifierModel = Union[LogisticRegression, RandomForestClassifier]
CONFIG_PATH = "src/config.yaml"


def create_logger(name: str, log_config: dict):
    logger = logging.getLogger(name)
    logger.setLevel(log_config['level'])
    formatter = logging.Formatter(
        fmt=log_config['format'],
        datefmt=log_config['date_format']
    )
    ch = logging.StreamHandler()
    ch.setLevel(log_config['level'])
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def load_config():
    path = os.getenv('CONFIG_PATH') or CONFIG_PATH
    with open(path) as fin:
        config = yaml.safe_load(fin)
    return config


def load_object(path: str):
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    feature_names: List[str]
    

class ModelResponse(BaseModel):
    disease: int


def prediction(
    data: List,
    feature_names: List[str],
    model: SkleratnClassifierModel,
    transformer: ColumnTransformer,
) -> List[ModelResponse]:
    data = pd.DataFrame(data, columns=feature_names)
    transfromed_data = pd.DataFrame(transformer.transform(data))
    predictions = model.predict(transfromed_data)
    return [ModelResponse(disease=predictions)]
