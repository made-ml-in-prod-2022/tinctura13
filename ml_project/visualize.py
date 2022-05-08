import os

import hydra
import pandas as pd
from omegaconf import DictConfig
from pandas_profiling import ProfileReport
from src.configs.train_params import (TrainingPipelineParams,
                                      TrainingPipelineParamsSchema)


def make_report(params: TrainingPipelineParams):
    data = pd.read_csv(params.input_data_path)
    ProfileReport(
        data,
        title="Exploratory data analysis report"
    ).to_file(params.report_path)
    
    
@hydra.main(config_path="configs", config_name="train_config")
def make_report_command(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    make_report(params)
    
    
if __name__ == "__main__":
    make_report_command()
