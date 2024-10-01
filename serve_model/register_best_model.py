import mlflow
from mlflow.tracking import MlflowClient
import os

def register_best_model(experiment_name, metric):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    best_run = None
    best_metric_value = -float('inf')

    runs = client.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.{metric} DESC"])

    if len(runs) > 0:
        best_run = runs[-1]  # Best run based on the chosen metric
        print(f"Best run ID: {best_run.info.run_id}, {metric}: {best_run.data.metrics[metric]}")
    else:
        print("No runs found.")

    model_uri = ""

    if best_run:
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model_name = "deployment"

        # Register model
        mlflow.register_model(model_uri, model_name)

        current_version_info = client.get_latest_versions(model_name, stages=["None"])

        latest_version_info = client.get_latest_versions(model_name, stages=["Production"])

        new_version = current_version_info[-1].version
        if latest_version_info:
            latest_version = latest_version_info[0].version  # Assuming the first one is the latest

            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Archived"
            )
            print(f"Previous Production model version {latest_version.version} archived")

        client.transition_model_version_stage(
            name=model_name,
            version=new_version,
            stage="Production"
        )
        print(f"Model version {new_version} transitioned to 'Production'")

    return model_uri

def build_run(model_uri):
    os.system(f"mlflow models build-docker -m '{model_uri}' -n 'deployment_image'")

if __name__ == "__main__":
    model_uri = register_best_model("/wine-quality", "eval_rmse")
    build_run(model_uri)