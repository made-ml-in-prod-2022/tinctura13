"""Super cool API to predict Heart Disease"""
import os
from typing import List

import uvicorn
from fastapi import FastAPI

from src.utils.utils import (HeartDiseaseModel, ModelResponse, create_logger,
                             load_config, load_object, prediction)

app = FastAPI()
config = load_config()
logger = create_logger('app', config['logging'])
model = None
transformer = None

@app.on_event('startup')
def load_model():
    """Loads model at startup of the app"""
    logger.info('Loading model...')
    model_path = config['model_path']
    transformer_path = config['transformer_path']
    if not model_path or not transformer_path:
        error = (
            f'MODEL_PATH is {model_path}, TRANSFORMER_PATH is {transformer_path}'
        )
        logger.error(error)
        raise RuntimeError(error)
    global model, transformer
    model = load_object(model_path)
    transformer = load_object(transformer_path)
    logger.info(f'Model loading completed')
    
    
@app.get('/')
async def root():
    return {'message': 'Welcome to Heart Disease classificator. Check /docs'}


@app.get('/predict/', response_model=List[ModelResponse])
async def predict(request: HeartDiseaseModel):
    logger.info("Making predictions...")
    return prediction(request.data, request.feature_names, model, transformer)
    
    
@app.get('/health')
async def health_check():
    if model:
        return 200


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=os.getenv("PORT", 8099)
    )
