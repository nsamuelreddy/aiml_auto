from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


REGRESSION_MODELS = {

    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        random_state=42
    ),

    "KNN Regressor": KNeighborsRegressor(),

    "SVR": SVR(),

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
def train_regression_models(
    X_train,
    y_train,
    X_test,
    progress_callback=None
):

    trained_models = {}
    predictions = {}

    total_models = len(REGRESSION_MODELS)

    for index, (model_name, model) in enumerate(REGRESSION_MODELS.items(), start=1):

        if progress_callback is not None:
            progress_callback(index - 1, total_models, model_name)

        model.fit(X_train, y_train)

        trained_models[model_name] = model

        predictions[model_name] = model.predict(X_test)

        if progress_callback is not None:
            progress_callback(index, total_models, model_name)

    return trained_models, predictions