from dataclasses import dataclass

from marshmallow_dataclass import class_schema


@dataclass()
class InferencePipelineParams:
    input_data_path: str
    predictions_path: str
    transformer_path: str
    model_path: str
    
    
InferencePipelineParamsSchema = class_schema(InferencePipelineParams)
