from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow.data
import mlflow
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.models import infer_signature


def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y, signature, dataset):
    # Define model architecture
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)

    model = Ridge(alpha=params["alpha"], solver='auto')


    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        mlflow.log_input(dataset, context="training")

        model.fit(train_x, train_y)

        valid_preds = model.predict(valid_x)

        eval_rmse = mean_squared_error(valid_y, valid_preds, squared=False)

        # Log parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        mlflow.sklearn.log_model(model, "model", signature=signature)

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

def objective(params, train_x, train_y, valid_x, valid_y, test_x, test_y, signature, dataset):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
        signature= signature,
        dataset= dataset
    )
    return result


def train():
    dataset_source_url = "../data/winequality-white.csv"
    data = pd.read_csv(
        dataset_source_url,
        sep=";",
    )

    # Split the data into training, validation, and test sets
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1).values
    train_y = train[["quality"]].values.ravel()
    test_x = test.drop(["quality"], axis=1).values
    test_y = test[["quality"]].values.ravel()
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42
    )
    signature = infer_signature(train_x, train_y)
    dataset: PandasDataset = from_pandas(data, source=dataset_source_url)

    space = {
        "alpha": hp.loguniform("alpha", np.log(1e-3), np.log(10))  # Ridge regularization strength
    }

    # remote_server_uri = "http://localhost:8080"  # set to your server URI
    # mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment("/wine-quality")

    with mlflow.start_run():
        # Conduct the hyperparameter search using Hyperopt
        trials = Trials()
        best = fmin(
            fn= lambda args: objective(args, train_x, train_y, valid_x, valid_y, test_x, test_y, signature, dataset),
            space=space,
            algo=tpe.suggest,
            max_evals=8,
            trials=trials,
        )

        # Fetch the details of the best run
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

        # Log the best parameters, loss, and model
        mlflow.log_params(best)
        mlflow.log_metric("eval_rmse", best_run["loss"])
        mlflow.sklearn.log_model(best_run["model"], "model", signature=signature)

        # Print out the best parameters and corresponding loss
        print(f"Best parameters: {best}")
        print(f"Best eval rmse: {best_run['loss']}")


if __name__ == "__main__":
    train()