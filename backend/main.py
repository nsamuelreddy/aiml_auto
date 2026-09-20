import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.comparison.model_comparison import best_model, compare_models, top_models
from app.comparison.regression_comparison import (
    best_model as best_regression_model,
    compare_regression_models,
    top_models as top_regression_models,
)
from app.evaluation.model_evaluation import evaluate_models
from app.evaluation.regression_evaluation import evaluate_regression_models
from app.models.model_training import train_models
from app.models.regression_training import train_regression_models
from app.modules.dataset_analyzer import dataset_shape, load_dataset, missing_values
from app.preprocessing.clean_data import (
    clean_column_names,
    drop_irrelevant_columns,
    remove_duplicate_rows,
    remove_empty_columns,
    remove_empty_rows,
)
from app.preprocessing.encoding import encoding_report, label_encode, one_hot_encoding
from app.preprocessing.feature_selection import (
    correlation_selection,
    feature_selection_report,
    variance_threshold_selection,
)
from app.preprocessing.missing_values import (
    fill_categorical_missing,
    fill_numerical_missing,
    missing_value_report,
)
from app.preprocessing.scaling import standard_scale
from app.preprocessing.split_data import split_dataset
from app.tuning.hyperparameter_tuner import screen_and_tune

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if pd.isna(value):
        return None
    return value


def _infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def _encode_target_labels(target: pd.Series) -> tuple[pd.Series, dict[str, int] | None]:
    cleaned_target = target.copy()

    if cleaned_target.empty:
        raise ValueError("Target column is empty. Please select a valid target column.")

    if pd.api.types.is_numeric_dtype(cleaned_target):
        return cleaned_target.astype(float), None

    cleaned_target = cleaned_target.astype(str).str.strip()
    cleaned_target = cleaned_target.replace({"nan": np.nan, "None": np.nan, "null": np.nan})

    unique_labels = sorted(cleaned_target.dropna().unique().tolist())
    if not unique_labels:
        raise ValueError("Target column has no valid labels after cleaning. Please choose a non-empty target column.")

    if len(unique_labels) == 2:
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = cleaned_target.map(mapping)
        encoded = encoded.astype("Int64")
        return encoded, {str(k): int(v) for k, v in mapping.items()}

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    valid_target = cleaned_target.dropna()
    encoded_values = le.fit_transform(valid_target)
    encoded = pd.Series(np.nan, index=cleaned_target.index, dtype=float)
    encoded.loc[valid_target.index] = encoded_values.astype(float)
    encoded = encoded.astype("Int64")
    mapping = {str(label): int(index) for index, label in enumerate(le.classes_)}
    return encoded, mapping


def _serialize_evaluation_report(evaluation_report: dict[str, Any], problem_type: str) -> dict[str, Any]:
    serializable_evaluation: dict[str, Any] = {}

    for model_name, metrics in evaluation_report.items():
        serializable_metrics: dict[str, Any] = {}
        for metric_name, metric_value in metrics.items():
            if problem_type == "classification" and metric_name == "Classification Report":
                serializable_metrics[metric_name] = str(metric_value)
            elif problem_type == "classification" and metric_name == "Confusion Matrix":
                serializable_metrics[metric_name] = metric_value.tolist()
            else:
                serializable_metrics[metric_name] = _to_serializable(metric_value)

        serializable_evaluation[model_name] = serializable_metrics

    return serializable_evaluation


def _emit_progress(progress_callback, progress: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(progress, message)


def _build_feature_importance_report(
    trained_models: dict[str, Any],
    feature_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {}

    for model_name, model in trained_models.items():
        importances = None

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(getattr(model, "coef_"), dtype=float)
            if coefficients.ndim == 1:
                importances = np.abs(coefficients)
            elif coefficients.ndim == 2:
                importances = np.abs(coefficients).mean(axis=0)

        if importances is None or importances.size == 0:
            continue

        if importances.shape[0] != len(feature_names):
            continue

        normalized_importances = importances / importances.sum() if importances.sum() else importances
        ranked_features = sorted(
            zip(feature_names, normalized_importances, strict=False),
            key=lambda item: item[1],
            reverse=True,
        )

        report[model_name] = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in ranked_features[:10]
            if float(importance) > 0
        ]

    return report


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_pipeline_result(
    job_id: str,
    file_path: str,
    target_column: str,
    progress_callback=None,
    skip_svm_by_size: bool = False,
    selected_model_name: str | None = None,
) -> dict[str, Any]:
    _emit_progress(progress_callback, 2, "Loading dataset")
    original_df = load_dataset(file_path)

    # In production (Render free tier), downsample large datasets to 10,000 rows
    # to stay safely within the 512MB RAM limit and prevent OOM restarts.
    if os.environ.get("RENDER") == "true" and len(original_df) > 10000:
        original_df = original_df.sample(n=10000, random_state=42).reset_index(drop=True)

    _emit_progress(progress_callback, 6, "Cleaning column names")
    df = clean_column_names(original_df.copy())

    # Normalize the chosen target before any feature cleaning. This prevents the label from being removed
    # by irrelevant-column filtering or one-hot encoding later in the pipeline.
    target_column = target_column.strip().lower().replace(" ", "_")
    normalized_to_original = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    if target_column not in df.columns and target_column in normalized_to_original:
        target_column = normalized_to_original[target_column]

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found after preprocessing. "
            "Please choose a valid target column."
        )

    y = df[target_column].copy()
    df = df.drop(columns=[target_column])

    y, target_mapping = _encode_target_labels(y)
    valid_mask = y.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    _emit_progress(progress_callback, 10, "Removing duplicate rows")
    duplicate_mask = ~df.duplicated()
    duplicate_rows = int((~duplicate_mask).sum())
    df = df.loc[duplicate_mask].reset_index(drop=True)
    y = y.loc[duplicate_mask].reset_index(drop=True)

    _emit_progress(progress_callback, 14, "Removing empty rows")
    empty_row_mask = ~df.isna().all(axis=1)
    df = df.loc[empty_row_mask].reset_index(drop=True)
    y = y.loc[empty_row_mask].reset_index(drop=True)

    _emit_progress(progress_callback, 18, "Removing empty columns")
    df = remove_empty_columns(df)

    _emit_progress(progress_callback, 22, "Dropping irrelevant columns")
    df, dropped_columns = drop_irrelevant_columns(df)

    _emit_progress(progress_callback, 28, "Filling missing values")
    df, numerical_report = fill_numerical_missing(df)
    df, categorical_report = fill_categorical_missing(df)
    missing_report = missing_value_report(numerical_report, categorical_report)

    _emit_progress(progress_callback, 36, "Encoding categorical features")
    df, label_report, encoder_artifact = label_encode(df)
    df, one_hot_report = one_hot_encoding(df)
    encoding_final_report = encoding_report(label_report, one_hot_report)

    _emit_progress(progress_callback, 46, "Preparing features")
    x = df
    problem_type = _infer_problem_type(y)
    input_feature_names = x.columns.tolist()

    _emit_progress(progress_callback, 54, "Scaling features")
    x, scaling_report, scaler = standard_scale(x)

    _emit_progress(progress_callback, 60, "Selecting features")
    x, variance_report = variance_threshold_selection(x)
    x, correlation_report = correlation_selection(x)
    feature_report = feature_selection_report(variance_report, correlation_report)
    selected_feature_names = x.columns.tolist()

    warnings: list[str] = []
    try:
        _emit_progress(progress_callback, 68, "Splitting train and test data")
        X_train, X_test, y_train, y_test = split_dataset(x, y, stratify=problem_type == "classification")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"Stratified split failed, using a random split instead: {str(exc)}")
        X_train, X_test, y_train, y_test = split_dataset(x, y, stratify=False)

    # Callbacks are identical for both problem types — define once
    def training_progress(completed: int, total: int, name: str) -> None:
        progress = 70 + int((completed / max(total, 1)) * 12)
        _emit_progress(progress_callback, progress, f"Training model: {name}")

    def tuning_progress(message: str) -> None:
        _emit_progress(progress_callback, 85, message)

    # Train all models
    if problem_type == "classification":
        _emit_progress(progress_callback, 70, "Training classification models")
        trained_models, _ = train_models(
            X_train, y_train, X_test,
            progress_callback=training_progress,
            skip_svm_by_size=skip_svm_by_size,
            selected_model_name=selected_model_name,
        )
    else:
        _emit_progress(progress_callback, 70, "Training regression models")
        trained_models, _ = train_regression_models(
            X_train, y_train, X_test,
            progress_callback=training_progress,
            skip_svr_by_size=skip_svm_by_size,
            selected_model_name=selected_model_name,
        )

    # Tune top 2 — same for both problem types
    if selected_model_name is not None and selected_model_name in trained_models:
        _emit_progress(progress_callback, 83, f"Tuning selected model: {selected_model_name}")
        trained_models, tuning_summary = screen_and_tune(
            trained_models={selected_model_name: trained_models[selected_model_name]},
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            progress_callback=tuning_progress,
            top_n=1,
        )
    else:
        _emit_progress(progress_callback, 83, "Screening top 2 for hyperparameter tuning")
        trained_models, tuning_summary = screen_and_tune(
            trained_models=trained_models,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            progress_callback=tuning_progress,
            top_n=2,
        )

    # Re-generate predictions from (possibly) tuned models, then evaluate
    _emit_progress(progress_callback, 88, "Evaluating tuned models")
    predictions = {name: model.predict(X_test) for name, model in trained_models.items()}

    if problem_type == "classification":
        evaluation_report = evaluate_models(trained_models, predictions, X_test, y_test)
        comparison = compare_models(evaluation_report)
        top3 = top_models(comparison, top_n=3)
        best = best_model(comparison)
        metric_options = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        primary_metric = "Accuracy"
        leaderboard_metrics = ["Accuracy", "F1 Score"]
    else:
        evaluation_report = evaluate_regression_models(trained_models, predictions, y_test)
        comparison = compare_regression_models(evaluation_report)
        top3 = top_regression_models(comparison, top_n=3)
        best = best_regression_model(comparison)
        metric_options = ["R2", "RMSE", "MAE", "MSE"]
        primary_metric = "R2"
        leaderboard_metrics = ["R2", "RMSE"]

    _emit_progress(progress_callback, 92, "Comparing models")
    best_model_name = str(best["Model"])
    best_model_object = trained_models.get(best_model_name)

    serializable_evaluation = _serialize_evaluation_report(evaluation_report, problem_type)
    best_model_summary = _to_serializable(best.to_dict() if hasattr(best, "to_dict") else best)
    feature_importance = _build_feature_importance_report(trained_models, selected_feature_names)

    best_metric_value = best_model_summary.get(primary_metric)
    dashboard_summary = {
        "problem_type": problem_type.title(),
        "rows": int(original_df.shape[0]),
        "columns": int(original_df.shape[1]),
        "missing_values": int(original_df.isna().sum().sum()),
        "best_model": best_model_name,
        "best_metric_label": primary_metric,
        "best_metric_value": _safe_float(best_metric_value),
    }

    _emit_progress(progress_callback, 100, "Pipeline completed")

    return {
        "job_id": job_id,
        "problem_type": problem_type,
        "metric_options": metric_options,
        "primary_metric": primary_metric,
        "leaderboard_metrics": leaderboard_metrics,
        "dataset": {
            "rows": int(original_df.shape[0]),
            "columns": int(original_df.shape[1]),
            "original_shape": dataset_shape(original_df),
            "missing_values": _to_serializable(missing_values(original_df)),
            "missing_values_total": int(original_df.isna().sum().sum()),
            "column_names": df.columns.tolist(),
            "dropped_columns": dropped_columns,
            "duplicate_rows_removed": int(duplicate_rows),
        },
        "preprocessing": {
            "missing_value_report": _to_serializable(missing_report),
            "encoding_report": _to_serializable(encoding_final_report),
            "feature_report": _to_serializable(feature_report),
            "scaling_report": _to_serializable(scaling_report),
            "warnings": warnings,
        },
        "evaluation": serializable_evaluation,
        "comparison": comparison.to_dict(orient="records"),
        "top3": top3.to_dict(orient="records"),
        "best_model": best_model_summary,
        "best_model_name": best_model_name,
        "best_model_object": best_model_object,
        "feature_names": input_feature_names,
        "selected_feature_names": selected_feature_names,
        "feature_importance": feature_importance,
        "dashboard_summary": dashboard_summary,
        "target_column": target_column,
        "target_label_mapping": target_mapping,
        "tuning_summary": _to_serializable(tuning_summary),
    }


