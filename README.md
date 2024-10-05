This Repository contains various MLFlow demo projects.

## ML Model CI pipeline - Dagster Pipeline to train and evaluate ML model

This project trains a machine learning model using the XGBoostRegressor and logs it to the MLFlow server. Upon each code push to the repository, a workflow is triggered that initiates a Dagster pipeline, which trains the model, logs it on a MLFlow Server, evaluates the model, and only approves the changes if the Mean Absolute Error (MAE) is below the specified ```max_mae_threshold```.

## Experiment Tracking with MLFlow and Serving Model on Kubernetes
[Serve Model](https://github.com/supreetshm947/MLFlowDemo/edit/master/serve_model)
