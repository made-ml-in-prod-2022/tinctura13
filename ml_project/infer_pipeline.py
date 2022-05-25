import logging
import os
from typing import NoReturn

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.configs.infer_params import (InferencePipelineParams,
                                      InferencePipelineParamsSchema)
from src.features.build_features import deserialize_transformer, make_features
from src.models.predict_model import predict_model
from src.utils.utils import deserialize_model

logger = logging.getLogger("infer_pipeline")
formatter = logging.Formatter("%(asctime)s %(user)-8s %(levelname)s: %(message)s")
logger.setLevel(logging.INFO)


def infer_pipeline(params: InferencePipelineParams) -> NoReturn:
    logger.info(f"inference pipeline with parameters: {params}")
    data = pd.read_csv(params.input_data_path)
    logger.info(f"dataset shape is {data.shape}")
    transformer = deserialize_transformer(params.transformer_path)
    transformed_data = make_features(transformer, data)
    logger.info(f"transformed data shape is: {transformed_data.shape}")
    
    model = deserialize_model(params.model_path)
    predictions, _ = predict_model(model, transformed_data)
    logger.info(f"predictions shape is: {predictions.shape}")
    pd.DataFrame(predictions).to_csv(params.predictions_path, header=False)
    logger.info(f"predictions saved to {params.predictions_path}")


@hydra.main(config_path="configs", config_name="infer_config")
def infer_pipeline_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = InferencePipelineParamsSchema()
    params = schema.load(cfg)
    infer_pipeline(params)
    

if __name__ == "__main__":
    infer_pipeline_command()
