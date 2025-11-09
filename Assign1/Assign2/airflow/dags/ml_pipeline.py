# airflow/dags/ml_pipeline.py
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

FEATURE_STORE = "datamart/gold/feature_store"
LABEL_STORE   = "datamart/gold/label_store"
MODEL_STORE   = "datamart/gold/model_store"
METRICS_STORE = "datamart/gold/metrics_store"
PRED_STORE    = "datamart/gold/predictions_store"

TRAIN_START = "2023-01-01"
TRAIN_END   = "2023-09-01" 

default_args = {
    "owner": "you",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="ml_monthly_pipeline",
    start_date=datetime(2023, 7, 1),
    schedule_interval="0 2 1 * *",
    catchup=False,
    default_args=default_args,
    tags=["ml", "spark"],
) as dag:


    train = BashOperator(
        task_id="train_models",
        bash_command=(
            "python -m utils.train_models "
            f"--feature-store {FEATURE_STORE} "
            f"--label-store {LABEL_STORE} "
            f"--start {TRAIN_START} --end {TRAIN_END} "
            f"--model-store {MODEL_STORE} "
            f"--metrics-out {METRICS_STORE}"
        ),
        cwd="/app",
    )

    infer = BashOperator(
        task_id="run_inference",
        bash_command=(
            "python -m utils.run_inference "
            f"--feature-store {FEATURE_STORE} "
            f"--model-store {MODEL_STORE} "
            "--start {{ ds }} --end {{ ds }} "
            f"--pred-out {PRED_STORE}"
        ),
        cwd="/app",
    )

    train >> infer
