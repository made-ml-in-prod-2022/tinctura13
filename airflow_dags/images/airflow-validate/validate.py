import os
import pickle
import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow
import logging

@click.command("validate")
@click.option("--data-dir")
@click.option("--transformer-dir")
@click.option("--model-name")
def validate(data_dir: str, transformer_dir: str, model_name: str):
    logger = logging.getLogger("evaluate_model")
    
    with open(os.path.join(transformer_dir, "scaler.okl"), "rb") as fin:
        scaler = pickle.load(fin)
    val = pd.read_csv(os.path.join(data_dir, "val.csv"), index_col=0)
    val_X, val_y = val.drop("target", 1), val["target"]
    val_X = scaler.transform(val_X)
    
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URL"])
    client = mlflow.tracking.MlflowClient()
    last_model_data = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda x: x.last_updated_timestamp,
        reverse=True,
    )[0]
    logger.warning("warn message")
    logger.warning(last_model_data)
    logger.warning(f"{last_model_data.name}, {last_model_data.version}")
    logger.warning(os.getcwd())
    logger.warning(os.listdir(path="."))
    model = mlflow.sklearn.load_model(
        f"model:/{last_model_data.name}/{str(last_model_data.version)}"
    )
    
    mlflow.set_experiment(f"validate_{model_name}")
    with mlflow.start_run():
        predictions = model.predict(val_X)
        mlflow.log_metric("roc_auc_score", roc_auc_score(val_y, predictions))
        mlflow.log_metric("accuracy_score", accuracy_score(val_y, predictions))
        mlflow.log_metric("f1_score", f1_score(val_y, predictions))
    
    client.transition_model_version_stage(
        name=last_model_data.name,
        version=last_model_data.version,
        stage="Production",
    )


if __name__ == "__main__":
    validate()
