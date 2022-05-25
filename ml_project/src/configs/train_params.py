from dataclasses import dataclass, field
from typing import List, Union

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class LRParams:
    model_type: str = field(default="LogisticRegression")
    penalty: str = field(default="l2")
    tol: float = field(default=1e-4)
    C: float = field(default=1.0)
    random_state: int = field(default=118)
    

@dataclass()
class RFParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=200)
    criterion: str = field(default="gini")
    random_state: int = field(default=118)
    

@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    report_path: str
    # features
    categorical_features: List[str]
    numerical_features: List[str]
    target_column: str
    # split
    val_size: float
    random_state: int
    # train
    model: Union[LRParams, RFParams]
    
    
TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))    
