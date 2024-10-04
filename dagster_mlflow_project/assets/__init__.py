from dagster import asset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

@asset
def run():
    np.random.seed(40)
    data = pd.read_csv("data/winequality-white.csv", sep=";")

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_x = train.drop(["quality"], axis=1).values
    train_y = train[["quality"]].values.ravel()
    test_x = test.drop(["quality"], axis=1).values
    test_y = test[["quality"]].values.ravel()

    signature = infer_signature(train_x, train_y)

    learning_rate = 0.01

    with mlflow.start_run():
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=learning_rate, random_state=42)
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("XGBoost model (learning_rate=%f):" % learning_rate)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", learning_rate)
        # mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        print("Metrics logged")

        mlflow.sklearn.log_model(model, "sk_models")
        print("Model training complete")
        return True