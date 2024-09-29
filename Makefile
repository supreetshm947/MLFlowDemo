start:
	echo "Starting MLFlow server"
	export MLFLOW_TRACKING_URI=http://localhost:8080
	mlflow server --host 127.0.0.1 --port 8080