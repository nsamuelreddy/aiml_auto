from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np


def evaluate_regression_models(
    trained_models,
    predictions,
    y_test
):

    evaluation_report = {}

    for model_name, model in trained_models.items():

        y_pred = predictions[model_name]

        mae = mean_absolute_error(y_test, y_pred)

        mse = mean_squared_error(y_test, y_pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, y_pred)

        evaluation_report[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    return evaluation_report
