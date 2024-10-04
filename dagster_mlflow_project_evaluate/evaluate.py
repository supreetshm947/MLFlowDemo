import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys


def evaluate():
    max_mae_threshold = 1.2
    experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

    client = MlflowClient()

    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)

    # Get the latest run ID
    latest_run_id = runs[0].info.run_id

    # Load the latest logged model from this run
    model_uri = f"runs:/{latest_run_id}/sk_models"

    model = mlflow.sklearn.load_model(model_uri)

    data = pd.read_csv("data/winequality-white.csv", sep=";")

    sample_data = data.sample(n=5, random_state=42)

    X_sample = sample_data.drop(columns=["quality"])  # Replace 'quality' with your actual label column if different
    y_true = sample_data["quality"]

    y_pred = pd.Series(model.predict(X_sample))

    mae = mean_absolute_error(y_true, y_pred)

    if mae > max_mae_threshold:
        print(f"MAE {mae} exceeds the threshold of {max_mae_threshold}. Failing the workflow.")
        sys.exit(1)
    print(f"Model passed the evaluation.")

if __name__ == "__main__":
    evaluate()