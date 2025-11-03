from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
sys.path.append('/opt/airflow')

from utils.data_processing import DataProcessor
from utils.ml_training import MLTrainer
from utils.ml_inference import MLInference
from utils.ml_monitoring import MLMonitor

# default args
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'loan_default_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for loan default prediction',
    schedule_interval='@monthly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'loan', 'default_prediction'],
)

# Task functions

def create_bronze_layer_task(**context):
    """Bronze layer: ingest raw data"""
    processor = DataProcessor(
        data_path='/opt/airflow/data',
        output_path='/opt/airflow/datamart'
    )
    processor.create_bronze_layer()
    print("Bronze layer completed")

def create_silver_layer_task(**context):
    """Silver layer: clean and validate data"""
    processor = DataProcessor(
        data_path='/opt/airflow/data',
        output_path='/opt/airflow/datamart'
    )
    processor.create_silver_layer()
    print("Silver layer completed")

def create_gold_layer_task(**context):
    """Gold layer: create ML-ready features"""
    processor = DataProcessor(
        data_path='/opt/airflow/data',
        output_path='/opt/airflow/datamart'
    )
    processor.create_gold_layer()
    print("Gold layer completed")

def train_model_task(**context):
    """Train ML models and select best one"""
    trainer = MLTrainer(model_path='/opt/airflow/models')
    trainer.train_models(feature_path='/opt/airflow/datamart/gold/feature_store.parquet')
    print("Model training completed")

def make_predictions_task(**context):
    """Make predictions using trained model"""
    inference = MLInference(model_path='/opt/airflow/models')
    inference.predict(
        feature_path='/opt/airflow/datamart/gold/feature_store_with_ids.parquet',
        output_path='/opt/airflow/datamart/gold/predictions.parquet'
    )
    print("Predictions completed")

def monitor_model_task(**context):
    """Monitor model performance and data drift"""
    monitor = MLMonitor(output_path='/opt/airflow/datamart/gold/monitoring')
    monitor.generate_monitoring_report(
        predictions_path='/opt/airflow/datamart/gold/predictions.parquet',
        labels_path='/opt/airflow/datamart/gold/label_store.parquet',
        reference_data_path='/opt/airflow/datamart/gold/feature_store.parquet'
    )
    print("Monitoring completed")

# Define tasks

bronze_task = PythonOperator(
    task_id='create_bronze_layer',
    python_callable=create_bronze_layer_task,
    dag=dag,
)

silver_task = PythonOperator(
    task_id='create_silver_layer',
    python_callable=create_silver_layer_task,
    dag=dag,
)

gold_task = PythonOperator(
    task_id='create_gold_layer',
    python_callable=create_gold_layer_task,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions_task,
    dag=dag,
)

monitor_task = PythonOperator(
    task_id='monitor_model',
    python_callable=monitor_model_task,
    dag=dag,
)

# Set task dependencies
# Data pipeline: bronze -> silver -> gold
bronze_task >> silver_task >> gold_task

# ML pipeline: gold -> train -> predict -> monitor
gold_task >> train_task >> predict_task >> monitor_task

