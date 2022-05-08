import json
import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.configs.train_params import (TrainingPipelineParams,
                                      TrainingPipelineParamsSchema)
from src.features.build_features import (build_transformer, make_features,
                                         serialize_transformer)
from src.models.predict_model import predict_model
from src.models.train_model import train_model
from src.utils.utils import evaluate_model, serialize_model

logger = logging.getLogger("train_pipeline")
logger.setLevel(logging.INFO)


def train_pipeline(params: TrainingPipelineParams):
    logger.info(f"training with params: {params}")
    data = pd.read_csv(params.input_data_path)
    logger.info(f"dataset shape is: {data.shape}")
    train, val = train_test_split(
        data,
        test_size=params.val_size,
        random_state=params.random_state
    )
    logger.info(f"train data shape is: {train.shape}")
    logger.info(f"validation data shape is: {val.shape}")
    
    train_features = train.drop(params.target_column, 1)
    train_target = train[params.target_column]
    transformer = build_transformer(params)
    transformer.fit(train_features)
    serialize_transformer(transformer, params.output_transformer_path)
    train_features = make_features(transformer, train_features)
    logger.info(f"train features shape is: {train_features.shape}")
    
    model = train_model(train_features, train_target, params.model)
    val_features = make_features(transformer, val)
    val_target = val[params.target_column]
    logging.info(f"validation features shape is: {val_features.shape}")
    predictions = predict_model(model, val_features)

    metrics = evaluate_model(predictions, val_target)
    with open(params.metric_path, "w") as metrics_file:
        json.dump(metrics, metrics_file)
    logger.info(f"metrics are: {metrics}")
    serialize_model(model, params.output_model_path)
    return metrics


@hydra.main(config_path="configs", config_name="train_config")
def train_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)
    
    
if __name__ == "__main__":
    train_pipeline_command()
