import pickle
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.configs.train_params import TrainingPipelineParams
from src.utils.scaler import CustomTransformer


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            ),
            (
                "ohe",
                OneHotEncoder()
            )
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline(
        [
            (
                "impute", 
                SimpleImputer(missing_values=np.nan, strategy="mean")
            ),
            (
                "scaler",
                CustomTransformer()
            )
        ]
    )
    return numerical_pipeline


def make_features(transformer: ColumnTransformer, data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(data))


def build_transformer(params: TrainingPipelineParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features
            ),
            (
                "numberical pipeline",
                build_numerical_pipeline(),
                params.numerical_features
            )
        ]
    )
    return transformer


def serialize_transformer(transformer: ColumnTransformer, output_file: str) -> NoReturn:
    with open(output_file, "wb") as fout:
        pickle.dump(transformer, fout)


def deserialize_transformer(input_file: str) -> ColumnTransformer:
    with open(input_file, "rb") as fin:
        transformer = pickle.load(fin)
    return transformer
