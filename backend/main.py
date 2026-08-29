import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any
import textwrap

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

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

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = ROOT_DIR / "saved_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Maximum allowed upload size in bytes (20 MB)
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024

ARTIFACT_FILENAMES = {
    "evaluation-report": "evaluation_report.csv",
    "model-comparison": "model_comparison.csv",
    "best-parameters": "best_parameters.json",
    "final-model": "final_model.pkl",
    "best-model": "best_model.pkl",
    "scaler": "scaler.pkl",
    "encoder": "encoder.pkl",
    "selected-features": "selected_features.pkl",
    "dashboard-pdf": "dashboard_report.pdf",
}

app = FastAPI(title="AutoML API", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT_DIR / "frontend"), name="static")

PIPELINE_JOBS: dict[str, dict[str, Any]] = {}
PIPELINE_JOBS_LOCK = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _update_pipeline_job(job_id: str, **updates: Any) -> None:
    with PIPELINE_JOBS_LOCK:
        job = PIPELINE_JOBS.setdefault(job_id, {"status": "queued", "progress": 0, "message": "Queued"})
        job.update(updates)


def _get_pipeline_job(job_id: str) -> dict[str, Any] | None:
    with PIPELINE_JOBS_LOCK:
        job = PIPELINE_JOBS.get(job_id)
        return dict(job) if job is not None else None


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


def _format_display_metric(value: Any, percent: bool = False) -> str:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "—"
    return f"{numeric_value * 100:.1f}%" if percent else f"{numeric_value:.3f}"


def _create_dashboard_pdf(job_id: str, result: dict[str, Any]) -> Path:
    artifact_dir = _artifact_directory(job_id)
    pdf_path = artifact_dir / ARTIFACT_FILENAMES["dashboard-pdf"]

    canvas = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        heading_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        heading_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    def draw_card(x: int, y: int, width: int, height: int, label: str, value: str) -> None:
        draw.rounded_rectangle((x, y, x + width, y + height), radius=22, fill="#f4f7fb", outline="#d7e2f0", width=2)
        draw.text((x + 24, y + 22), label, fill="#5b6b84", font=small_font)
        draw.text((x + 24, y + 58), value, fill="#0f172a", font=heading_font)

    def draw_wrapped_block(x: int, y: int, title: str, lines: list[str]) -> int:
        draw.text((x, y), title, fill="#0f172a", font=heading_font)
        cursor_y = y + 42
        for line in lines:
            wrapped = textwrap.wrap(line, width=95) or [""]
            for wrapped_line in wrapped:
                draw.text((x, cursor_y), wrapped_line, fill="#334155", font=body_font)
                cursor_y += 30
            cursor_y += 8
        return cursor_y

    problem_type = str(result.get("problem_type", "Unknown")).title()
    rows = str(result.get("dataset", {}).get("rows", "—"))
    columns = str(result.get("dataset", {}).get("columns", "—"))
    missing_values = str(result.get("dashboard_summary", {}).get("missing_values", result.get("dataset", {}).get("missing_values_total", "—")))
    best_model_name = str(result.get("best_model_name", result.get("best_model", {}).get("Model", "—")))
    metric_label = str(result.get("primary_metric", "Accuracy"))
    metric_value = result.get("dashboard_summary", {}).get("best_metric_value", result.get("best_model", {}).get(metric_label))
    percent_metric = result.get("problem_type") == "classification"

    draw.text((72, 58), "AutoML Dashboard Report", fill="#0f172a", font=title_font)
    draw.text((72, 112), f"Job ID: {job_id}", fill="#64748b", font=small_font)

    card_y = 180
    card_w = 340
    card_h = 120
    gap = 24
    draw_card(72, card_y, card_w, card_h, "Problem Type", problem_type)
    draw_card(72 + card_w + gap, card_y, card_w, card_h, "Rows", rows)
    draw_card(72 + (card_w + gap) * 2, card_y, card_w, card_h, "Columns", columns)
    draw_card(72, card_y + 146, card_w, card_h, "Missing Values", missing_values)
    draw_card(72 + card_w + gap, card_y + 146, card_w, card_h, "Best Model", best_model_name)
    draw_card(72 + (card_w + gap) * 2, card_y + 146, card_w, card_h, metric_label, _format_display_metric(metric_value, percent_metric))

    current_y = 460
    current_y = draw_wrapped_block(
        72,
        current_y,
        "Top Models",
        [
            f"1. {result.get('top3', [{}])[0].get('Model', '—')}",
            f"2. {result.get('top3', [{}, {}])[1].get('Model', '—')}" if len(result.get('top3', [])) > 1 else "2. —",
            f"3. {result.get('top3', [{}, {}, {}])[2].get('Model', '—')}" if len(result.get('top3', [])) > 2 else "3. —",
        ],
    )

    current_y += 24
    evaluation_section = []
    best_model = result.get("best_model", {})
    if isinstance(best_model, dict):
        for key in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "R2", "RMSE", "MAE", "MSE"]:
            if key in best_model:
                evaluation_section.append(f"{key}: {_format_display_metric(best_model[key], key in {'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'})}")

    if evaluation_section:
        current_y = draw_wrapped_block(72, current_y, "Best Model Metrics", evaluation_section)

    current_y += 20
    download_note = [
        "Saved artifacts included in this job:",
        "best_model.pkl, scaler.pkl, encoder.pkl, selected_features.pkl",
    ]
    draw_wrapped_block(72, current_y, "Artifacts", download_note)

    canvas.save(pdf_path, "PDF", resolution=150.0)
    return pdf_path


def _artifact_directory(job_id: str) -> Path:
    artifact_dir = ARTIFACTS_DIR / job_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _save_pipeline_artifacts(
    job_id: str,
    evaluation_report: dict[str, Any],
    comparison: pd.DataFrame,
    best_model_name: str,
    best_model_object: Any,
    scaler: Any,
    encoder_artifact: Any,
    input_feature_names: list[str],
    selected_feature_names: list[str],
    best_model_summary: dict[str, Any],
    problem_type: str,
) -> dict[str, str]:
    artifact_dir = _artifact_directory(job_id)

    evaluation_rows: list[dict[str, Any]] = []
    for model_name, metrics in evaluation_report.items():
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (list, tuple, dict)):
                serialized_value = json.dumps(_to_serializable(metric_value), default=str)
            else:
                serialized_value = metric_value

            evaluation_rows.append({
                "Model": model_name,
                "Metric": metric_name,
                "Value": serialized_value,
            })

    pd.DataFrame(evaluation_rows).to_csv(artifact_dir / ARTIFACT_FILENAMES["evaluation-report"], index=False)
    comparison.to_csv(artifact_dir / ARTIFACT_FILENAMES["model-comparison"], index=False)

    best_parameters_payload = {
        "problem_type": problem_type,
        "selected_model": best_model_name,
        "best_model": best_model_summary,
        "parameters": best_model_object.get_params(deep=False),
        "input_feature_names": input_feature_names,
        "selected_feature_names": selected_feature_names,
    }
    with (artifact_dir / ARTIFACT_FILENAMES["best-parameters"]).open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(best_parameters_payload), handle, indent=2, default=str)

    joblib.dump(best_model_object, artifact_dir / ARTIFACT_FILENAMES["final-model"])
    joblib.dump(best_model_object, artifact_dir / ARTIFACT_FILENAMES["best-model"])
    joblib.dump(scaler, artifact_dir / "scaler.pkl")
    joblib.dump(encoder_artifact, artifact_dir / "encoder.pkl")
    joblib.dump(input_feature_names, artifact_dir / "feature_names.pkl")
    joblib.dump(selected_feature_names, artifact_dir / "selected_feature_names.pkl")
    joblib.dump(selected_feature_names, artifact_dir / ARTIFACT_FILENAMES["selected-features"])

    return {
        "evaluation_report": f"/download/{job_id}/evaluation-report",
        "model_comparison": f"/download/{job_id}/model-comparison",
        "best_parameters": f"/download/{job_id}/best-parameters",
        "final_model": f"/download/{job_id}/final-model",
        "best_model": f"/download/{job_id}/best-model",
        "scaler": f"/download/{job_id}/scaler",
        "encoder": f"/download/{job_id}/encoder",
        "selected_features": f"/download/{job_id}/selected-features",
        "dashboard_pdf": f"/download/{job_id}/dashboard-pdf",
    }


def _load_job_artifacts(job_id: str) -> dict[str, Any]:
    artifact_dir = ARTIFACTS_DIR / job_id
    model_path = artifact_dir / ARTIFACT_FILENAMES["final-model"]
    feature_names_path = artifact_dir / "feature_names.pkl"
    selected_feature_names_path = artifact_dir / "selected_feature_names.pkl"
    scaler_path = artifact_dir / ARTIFACT_FILENAMES["scaler"]
    best_parameters_path = artifact_dir / ARTIFACT_FILENAMES["best-parameters"]

    if not model_path.exists() or not feature_names_path.exists() or not selected_feature_names_path.exists() or not scaler_path.exists():
        raise HTTPException(status_code=404, detail="Saved model artifacts not found for this job.")

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    selected_feature_names = joblib.load(selected_feature_names_path)
    scaler = joblib.load(scaler_path)
    best_parameters: dict[str, Any] | None = None
    if best_parameters_path.exists():
        with best_parameters_path.open("r", encoding="utf-8") as handle:
            best_parameters = json.load(handle)

    return {
        "model": model,
        "feature_names": feature_names,
        "selected_feature_names": selected_feature_names,
        "scaler": scaler,
        "best_parameters": best_parameters,
    }


def _build_pipeline_result(job_id: str, file_path: str, target_column: str, progress_callback=None) -> dict[str, Any]:
    _emit_progress(progress_callback, 2, "Loading dataset")
    original_df = load_dataset(file_path)

    # In production (Render free tier), downsample large datasets to 10,000 rows
    # to stay safely within the 512MB RAM limit and prevent OOM restarts.
    import os
    if os.environ.get("RENDER") == "true" and len(original_df) > 10000:
        original_df = original_df.sample(n=10000, random_state=42).reset_index(drop=True)

    _emit_progress(progress_callback, 6, "Cleaning column names")
    df = clean_column_names(original_df.copy())

    _emit_progress(progress_callback, 10, "Removing duplicate rows")
    df, duplicate_rows = remove_duplicate_rows(df)

    _emit_progress(progress_callback, 14, "Removing empty rows")
    df = remove_empty_rows(df)

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

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' was not found after preprocessing.")

    _emit_progress(progress_callback, 46, "Preparing features")
    y = df[target_column]
    x = df.drop(columns=[target_column])
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

    if problem_type == "classification":
        def training_progress(completed_models: int, total_models: int, model_name: str) -> None:
            progress = 70 + int((completed_models / max(total_models, 1)) * 16)
            _emit_progress(progress_callback, progress, f"Training model: {model_name}")

        _emit_progress(progress_callback, 70, "Training classification models")
        trained_models, predictions = train_models(X_train, y_train, X_test, progress_callback=training_progress)
        _emit_progress(progress_callback, 88, "Evaluating models")
        evaluation_report = evaluate_models(trained_models, predictions, X_test, y_test)
        _emit_progress(progress_callback, 92, "Comparing models")
        comparison = compare_models(evaluation_report)
        top3 = top_models(comparison, top_n=3)
        best = best_model(comparison)
        best_model_name = str(best["Model"])
        metric_options = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        primary_metric = "Accuracy"
        leaderboard_metrics = ["Accuracy", "F1 Score"]
    else:
        def training_progress(completed_models: int, total_models: int, model_name: str) -> None:
            progress = 70 + int((completed_models / max(total_models, 1)) * 16)
            _emit_progress(progress_callback, progress, f"Training model: {model_name}")

        _emit_progress(progress_callback, 70, "Training regression models")
        trained_models, predictions = train_regression_models(X_train, y_train, X_test, progress_callback=training_progress)
        _emit_progress(progress_callback, 88, "Evaluating models")
        evaluation_report = evaluate_regression_models(trained_models, predictions, y_test)
        _emit_progress(progress_callback, 92, "Comparing models")
        comparison = compare_regression_models(evaluation_report)
        top3 = top_regression_models(comparison, top_n=3)
        best = best_regression_model(comparison)
        best_model_name = str(best["Model"])
        metric_options = ["R2", "RMSE", "MAE", "MSE"]
        primary_metric = "R2"
        leaderboard_metrics = ["R2", "RMSE"]

    serializable_evaluation = _serialize_evaluation_report(evaluation_report, problem_type)
    best_model_summary = _to_serializable(best.to_dict() if hasattr(best, "to_dict") else best)
    best_model_object = trained_models[best_model_name]
    feature_importance = _build_feature_importance_report(trained_models, selected_feature_names)
    downloads = _save_pipeline_artifacts(
        job_id=job_id,
        evaluation_report=serializable_evaluation,
        comparison=comparison,
        best_model_name=best_model_name,
        best_model_object=best_model_object,
        scaler=scaler,
        encoder_artifact=encoder_artifact,
        input_feature_names=input_feature_names,
        selected_feature_names=selected_feature_names,
        best_model_summary=best_model_summary,
        problem_type=problem_type,
    )

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

    _create_dashboard_pdf(
        job_id,
        {
            "problem_type": problem_type,
            "dataset": dashboard_summary,
            "dashboard_summary": dashboard_summary,
            "best_model_name": best_model_name,
            "best_model": best_model_summary,
            "best_metric_value": best_metric_value,
            "primary_metric": primary_metric,
            "top3": top3.to_dict(orient="records"),
        },
    )
    downloads["dashboard_pdf"] = f"/download/{job_id}/dashboard-pdf"

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
        "feature_names": input_feature_names,
        "selected_feature_names": selected_feature_names,
        "feature_importance": feature_importance,
        "dashboard_summary": dashboard_summary,
        "downloads": downloads,
        "target_column": target_column,
    }


def _run_pipeline_job(job_id: str, file_path: str, target_column: str) -> None:
    def progress_callback(progress: int, message: str) -> None:
        _update_pipeline_job(job_id, status="running", progress=progress, message=message)

    try:
        _update_pipeline_job(job_id, status="running", progress=1, message="Starting pipeline")
        result = _build_pipeline_result(job_id, file_path, target_column, progress_callback=progress_callback)
        _update_pipeline_job(job_id, status="completed", progress=100, message="Pipeline completed", result=result)
    except Exception as exc:  # pragma: no cover - defensive API error handling
        _update_pipeline_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run-pipeline")
async def run_pipeline(file: UploadFile = File(...), target_column: str = Form("survived")) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A dataset file is required.")

    # Defensive server-side validation: reject files larger than MAX_UPLOAD_SIZE_BYTES
    # Try to determine size without consuming the upload stream if possible
    file_size = None
    try:
        # Seek to end to determine size for spooled/temp files
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
    except Exception:
        file_size = None

    if file_size is not None and file_size > MAX_UPLOAD_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File too large. Your file is {size_mb:.1f} MB. The maximum allowed size is 20 MB.")

    file_path = UPLOAD_DIR / file.filename

    # If size couldn't be determined earlier, stream to disk while enforcing the limit
    with file_path.open("wb") as buffer:
        total_written = 0
        chunk_size = 64 * 1024
        while True:
            chunk = file.file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            total_written += len(chunk)
            if total_written > MAX_UPLOAD_SIZE_BYTES:
                # remove partial file and return 413
                try:
                    buffer.close()
                except Exception:
                    pass
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass
                size_mb = total_written / (1024 * 1024)
                raise HTTPException(status_code=413, detail=f"File too large. Your file is {size_mb:.1f} MB. The maximum allowed size is 20 MB.")

    job_id = uuid.uuid4().hex
    _update_pipeline_job(job_id, status="queued", progress=0, message="Queued")

    worker = threading.Thread(target=_run_pipeline_job, args=(job_id, str(file_path), target_column))
    worker.start()

    return {"job_id": job_id, "status": "queued", "progress": 0, "message": "Queued"}


@app.get("/pipeline-status/{job_id}")
def pipeline_status(job_id: str) -> dict[str, Any]:
    job = _get_pipeline_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")

    return job


@app.get("/download/{job_id}/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str) -> FileResponse:
    filename = ARTIFACT_FILENAMES.get(artifact_name)
    if filename is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    artifact_path = _artifact_directory(job_id) / filename
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    media_type = "application/json" if artifact_name == "best-parameters" else "text/csv" if artifact_name in {"evaluation-report", "model-comparison"} else "application/octet-stream"
    return FileResponse(artifact_path, filename=filename, media_type=media_type)


@app.post("/predict/{job_id}")
def predict_with_saved_model(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = _load_job_artifacts(job_id)
    model = artifacts["model"]
    feature_names = list(artifacts["feature_names"])
    selected_feature_names = list(artifacts["selected_feature_names"])
    scaler = artifacts["scaler"]

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Prediction payload must be a JSON object.")

    raw_features = payload.get("features")
    if not isinstance(raw_features, dict):
        raise HTTPException(status_code=400, detail="'features' must be a JSON object.")

    missing_features = [name for name in feature_names if name not in raw_features]
    extra_features = [name for name in raw_features if name not in feature_names]
    if missing_features:
        raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing_features)}")
    if extra_features:
        raise HTTPException(status_code=400, detail=f"Unexpected features: {', '.join(extra_features)}")

    try:
        ordered_row = [{name: float(raw_features[name]) for name in feature_names}]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="All feature values must be numeric.")

    input_frame = pd.DataFrame(ordered_row, columns=feature_names)
    scaled_frame = pd.DataFrame(scaler.transform(input_frame), columns=feature_names)

    missing_selected = [name for name in selected_feature_names if name not in scaled_frame.columns]
    if missing_selected:
        raise HTTPException(status_code=400, detail=f"Prediction inputs do not contain required selected features: {', '.join(missing_selected)}")

    model_frame = scaled_frame[selected_feature_names]
    prediction = model.predict(model_frame)

    result: dict[str, Any] = {
        "job_id": job_id,
        "prediction": _to_serializable(prediction[0]),
        "features": ordered_row[0],
        "feature_names": feature_names,
        "selected_feature_names": selected_feature_names,
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(model_frame)[0]
            classes = getattr(model, "classes_", None)
            if classes is not None:
                result["probabilities"] = {
                    str(class_name): float(probability)
                    for class_name, probability in zip(classes, probabilities, strict=False)
                }
        except Exception:
            pass

    return result


@app.get("/")
def read_root() -> FileResponse:
    return FileResponse(ROOT_DIR / "frontend" / "index.html")
