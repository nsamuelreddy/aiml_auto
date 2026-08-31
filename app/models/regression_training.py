from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Skip SVR when training rows exceed this threshold (same rule as classification SVM).
SVR_ROW_LIMIT = 10_000

BASE_REGRESSION_MODELS = {

    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        random_state=42
    ),

    "KNN Regressor": KNeighborsRegressor(),

    "Gradient Boosting Regressor": GradientBoostingRegressor(
        random_state=42
    ),

    "XGBoost Regressor": XGBRegressor(
        random_state=42,
        eval_metric="rmse"
    ),

    "LightGBM Regressor": LGBMRegressor(
        random_state=42,
        verbose=-1
    )
}


def get_regression_models(skip_svr: bool = False) -> dict:
    models = dict(BASE_REGRESSION_MODELS)
    if not skip_svr:
        models["SVR"] = SVR()
    return models


def train_regression_models(
    X_train,
    y_train,
    X_test,
    progress_callback=None,
    skip_svr_by_size: bool = False,
):

    trained_models = {}
    predictions = {}

    n_rows = len(X_train)
    skip_svr = skip_svr_by_size or (n_rows > SVR_ROW_LIMIT)

    models = get_regression_models(skip_svr=skip_svr)

    total_models = len(models)

    for index, (model_name, model) in enumerate(models.items(), start=1):

        if progress_callback is not None:
            progress_callback(index - 1, total_models, model_name)

        model.fit(X_train, y_train)

        trained_models[model_name] = model

        predictions[model_name] = model.predict(X_test)

        if progress_callback is not None:
            progress_callback(index, total_models, model_name)

    if skip_svr and progress_callback is not None:
        if skip_svr_by_size:
            reason = "file size > 1.5 MB"
        else:
            reason = f"{n_rows:,} rows > {SVR_ROW_LIMIT:,} limit"
        progress_callback(total_models, total_models, f"SVR skipped ({reason})")

    return trained_models, predictions