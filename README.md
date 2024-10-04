This Repository contains various MLFlow demo projects.

## ML Model CI pipeline

This project trains a machine learning model using the XGBoostRegressor and logs it to the MLFlow server. Upon each code push to the repository, a workflow is triggered that builds the code, evaluates the model, and only approves the changes if the Mean Absolute Error (MAE) is below the specified ```max_mae_threshold```.

## Experiment Tracking with MLFlow and Serving Model on Kubernetes
[Serve Model](https://github.com/supreetshm947/MLFlowDemo/edit/master/serve_model)
