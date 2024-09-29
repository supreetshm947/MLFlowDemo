from mlflow import MlflowClient
client = MlflowClient()
client.restore_experiment("/wine-quality")
client.restore_experiment("/my-experiment")
