"""Parameter search spaces for the AutoML tuning pipeline."""

CLASSIFICATION_PARAMETER_GRIDS = {
    "Logistic Regression": {
        "C": [0.1, 1.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    },
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5],
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    },
    "KNN": {
        "n_neighbors": [3, 5],
        "weights": ["uniform", "distance"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
        "max_depth": [3, 5],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    },
}

REGRESSION_PARAMETER_GRIDS = {
    "Linear Regression": {
        "fit_intercept": [True, False],
    },
    "Decision Tree Regressor": {
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5],
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    },
    "KNN Regressor": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4],
    },
    "XGBoost Regressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "LightGBM Regressor": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [15, 31],
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    },
}

__all__ = [
    "CLASSIFICATION_PARAMETER_GRIDS",
    "REGRESSION_PARAMETER_GRIDS",
]
