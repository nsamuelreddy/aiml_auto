from pandas.api.types import is_numeric_dtype

from app.modules.dataset_analyzer import *

from app.preprocessing.clean_data import *
from app.preprocessing.missing_values import *
from app.preprocessing.encoding import *
from app.preprocessing.scaling import *
from app.preprocessing.feature_selection import *
from app.preprocessing.split_data import *
from app.comparison.model_comparison import *
from app.models.model_training import *
from backend.tuning import tune_model
from backend.parameter_grids import CLASSIFICATION_PARAMETER_GRIDS
from app.evaluation.model_evaluation import *
from app.models.regression_training import train_regression_models
from app.evaluation.regression_evaluation import evaluate_regression_models
from app.comparison.regression_comparison import (
    compare_regression_models,
    top_models as top_regression_models,
    best_model as best_regression_model
)
from backend.parameter_grids import REGRESSION_PARAMETER_GRIDS


def run_offline_pipeline(file_path: str = "datasets/bottle.csv"):
    # This is a copy of the previous top-level script preserved for manual runs.
    df = load_dataset(file_path)
    print(df.dtypes)

    df = clean_column_names(df)

    df, duplicate_rows = remove_duplicate_rows(df)

    df = remove_empty_rows(df)

    df = remove_empty_columns(df)

    df, dropped_columns = drop_irrelevant_columns(df)
    print("Dropped Columns:")
    print(dropped_columns)

    print("\nColumns after dropping:")
    print(df.columns.tolist())

    df, numerical_report = fill_numerical_missing(df)

    df, categorical_report = fill_categorical_missing(df)

    missing_report = missing_value_report(
        numerical_report,
        categorical_report
    )

    df, label_report = label_encode(df)

    df, one_hot_report = one_hot_encoding(df)

    encoding_final_report = encoding_report(
        label_report,
        one_hot_report
    )

    target_column = "salnty"

    y = df[target_column]

    x = df.drop(columns=[target_column])
    print("\nColumns in X:")
    print(x.columns.tolist())

    if is_numeric_dtype(y):

        if y.nunique() <= 20:
            problem_type = "classification"
        else:
            problem_type = "regression"

    else:
        problem_type = "classification"

    print(f"\nProblem Type: {problem_type}")

    x, scaling_report = standard_scale(x)

    x, variance_report = variance_threshold_selection(x)

    x, correlation_report = correlation_selection(x)

    feature_report = feature_selection_report(
        variance_report,
        correlation_report
    )

    X_train, X_test, y_train, y_test = split_dataset(
        x,
        y
    )

    if problem_type == "classification":

        trained_models, predictions = train_models(
            X_train,
            y_train,
            X_test
        )

    else:

        trained_models, predictions = train_regression_models(
            X_train,
            y_train,
            X_test
        )

    final_models = trained_models.copy()

    if problem_type == "classification":

        evaluation_report = evaluate_models(
            trained_models,
            predictions,
            X_test,
            y_test
        )

    else:

        evaluation_report = evaluate_regression_models(
            trained_models,
            predictions,
            X_test,
            y_test
        )

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    print()

    for model in trained_models.keys():
        print(model)

    print()

    print("Total Models Trained :", len(trained_models))

    print()

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print()

    for model_name, metrics in evaluation_report.items():

        print(model_name)
        print("-" * 50)

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        print()

    if problem_type == "classification":
        comparison = compare_models(evaluation_report)
    else:
        comparison = compare_regression_models(evaluation_report)

    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    print()

    if problem_type == "classification":
        top3 = top_models(comparison, top_n=3)
    else:
        top3 = top_regression_models(comparison, top_n=3)

    tuned_models = {}

    for model_name in top3["Model"]:

        model = trained_models[model_name]

        if problem_type == "classification":
            param_space = CLASSIFICATION_PARAMETER_GRIDS.get(model_name)
        else:
            param_space = REGRESSION_PARAMETER_GRIDS.get(model_name)

        if param_space is None:
            print(f"\nNo parameter grid found for {model_name}. Skipping tuning.")
            tuned_models[model_name] = model
            continue
        if problem_type == "classification":

            if model_name in ["Decision Tree", "KNN", "Naive Bayes"]:
                search_type = "grid"
            else:
                search_type = "random"

        else:

            if model_name in [
                "Linear Regression",
                "Decision Tree Regressor",
                "KNN Regressor"
            ]:
                search_type = "grid"
            else:
                search_type = "random"

        tuning_result = tune_model(
            model=model,
            param_space=param_space,
            x_train=X_train,
            y_train=y_train,
            search_type=search_type
        )

        tuned_models[model_name] = tuning_result["best_model"]

        print(f"\n{model_name}")
        print("Best Parameters:", tuning_result["best_params"])
        print("Best CV Score:", tuning_result["best_score"])

    tuned_predictions = {}

    for model_name, model in tuned_models.items():
        tuned_predictions[model_name] = model.predict(X_test)

    if problem_type == "classification":

        tuned_evaluation = evaluate_models(
            tuned_models,
            tuned_predictions,
            X_test,
            y_test
        )

    else:

        tuned_evaluation = evaluate_regression_models(
            tuned_models,
            tuned_predictions,
            X_test,
            y_test
        )

    final_evaluation = evaluation_report.copy()

    for model_name in tuned_models:

        if problem_type == "classification":
            original_metric = evaluation_report[model_name]["Accuracy"]
            tuned_metric = tuned_evaluation[model_name]["Accuracy"]
        else:
            original_metric = evaluation_report[model_name]["R2"]
            tuned_metric = tuned_evaluation[model_name]["R2"]

        if tuned_metric > original_metric:
            final_evaluation[model_name] = tuned_evaluation[model_name]
            final_models[model_name] = tuned_models[model_name]
            print(f"{model_name}: Using Tuned Model")
        else:
            print(f"{model_name}: Using Original Model")

    if problem_type == "classification":
        final_comparison = compare_models(final_evaluation)
    else:
        final_comparison = compare_regression_models(final_evaluation)

    if problem_type == "classification":
        final_top3 = top_models(final_comparison)
    else:
        final_top3 = top_regression_models(final_comparison)

    if problem_type == "classification":
        final_best = best_model(final_comparison)
    else:
        final_best = best_regression_model(final_comparison)

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print()
    print(final_comparison)

    print()

    print("=" * 70)
    print("TOP 3 MODELS")
    print("=" * 70)

    print()
    print(final_top3)

    print()

    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print()
    best_model_name = final_best["Model"]
    best_model_object = final_models[best_model_name]
    for key, value in final_best.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_offline_pipeline()
