from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from utils import DEFAULT_VOLUME, default_args

with DAG(
    "01_generate_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(3),
) as dag:
    start = DummyOperator(task_id="begin-generate-data")
    download_data = DockerOperator(
        task_id="docker-airflow-download",
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[DEFAULT_VOLUME],
    )

    end = DummyOperator(task_id="end-generate-data")

    start >> download_data >> end
