import os
from datetime import timedelta

default_args = {
    "owner": "airflow",
    "email_on_failure": True,
    "email": ["tinctura@gmail.com"], # "tinctura@gmail.com" "airflow@example.com"
    "retry_delay": timedelta(seconds=300),
    "retries": 1
}

mlflow_env = {"MLFLOW_TRACKING_URL": os.environ["MLFLOW_TRACKING_URL"]}
model_name = os.environ["MODEL_NAME"]

DEFAULT_VOLUME = "/home/tinctura/Documents/tinctura13/airflow_ml_dags/data:/data"
ARTIFACT_VOLUME = "mlrun_data:/mlruns"
