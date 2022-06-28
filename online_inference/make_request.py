import numpy as np
import pandas as pd
import requests

from src.features import CAT_FEATS, COL_ORDER, NUM_FEATS
from src.utils.utils import create_logger, load_config

NUM_ROWS_TO_GENERATE = 10


def generate_data(num_rows: int) -> pd.DataFrame:
    """Generates fake data for requests"""
    float_values = {}
    categorical_values = {}
    
    for feat, stat in NUM_FEATS.items():
        mu, sigma = stat.mean, stat.std
        float_values[feat] = np.random.normal(mu, sigma, num_rows)
    
    for feat, stat in CAT_FEATS.items():
        categorical_values[feat] = np.random.randint(0, stat.nunique, num_rows)
    
    generated_data = {**float_values, **categorical_values}
    return pd.DataFrame(generated_data)[COL_ORDER]


if __name__ == "__main__":
    config = load_config()
    logger = create_logger('request_generator', config['logging'])
    generated_data = generate_data(NUM_ROWS_TO_GENERATE)
    request_features = COL_ORDER
    
    for i in range(NUM_ROWS_TO_GENERATE):
        request_data = [_ for _ in generated_data.iloc[i].to_list()]
        logger.info(f'request_data: {request_data}')
        response = requests.get(
            f"http://{config['host']}:{config['port']}/predict/",
            json={"data": [request_data], "feature_names": request_features},
        )
        logger.info(f'status_code: {response.status_code}')
        logger.info(f'response.json:: {response.json()}')
