"""
Screen-then-Tune hyperparameter tuning for the top N models.

Strategy:
1. Screen  – rank all trained models by 3-fold CV score on the training data
             (avoids using the test set for model selection).
2. Select  – pick the top N models by CV score.
3. Tune    – run RandomizedSearchCV on each selected model.
4. Guard   – only replace the original model if tuning improves the CV score.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from backend.parameter_grids import (
    CLASSIFICATION_PARAMETER_GRIDS,
    REGRESSION_PARAMETER_GRIDS,
)

# Number of random parameter combinations tried per model.
# 4 iterations × 2 CV folds = 8 fits per model → fast and lightweight for cloud containers.
DEFAULT_N_ITER = 4
DEFAULT_CV = 2
DEFAULT_TOP_N = 2


def _cv_score(model: Any, X_train: Any, y_train: Any, scoring: str, cv: int) -> float:
    """Return mean cross-validation score for *model* on the training data."""
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score=0.0,
    )
    return float(scores.mean())


def screen_and_tune(
    trained_models: dict[str, Any],
    X_train: Any,
    y_train: Any,
    problem_type: str,
    top_n: int = DEFAULT_TOP_N,
    n_iter: int = DEFAULT_N_ITER,
    cv: int = DEFAULT_CV,
    progress_callback=None,
    baseline_scores: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Screen all trained models by CV score, tune the top N, return updated models
    and a summary of what happened.
    """
    is_classification = problem_type == "classification"
    scoring = "accuracy" if is_classification else "r2"
    param_grids = (
        CLASSIFICATION_PARAMETER_GRIDS if is_classification else REGRESSION_PARAMETER_GRIDS
    )

    # ── Step 1: Screen ────────────────────────────────────────────────────────
    if baseline_scores is None:
        if progress_callback:
            progress_callback("Screening models by validation score")
        baseline_scores = {}
        for model_name, model in trained_models.items():
            baseline_scores[model_name] = _cv_score(model, X_train, y_train, scoring, cv)
    else:
        baseline_scores = dict(baseline_scores)

    # ── Step 2: Select top N ──────────────────────────────────────────────────
    ranked = sorted(baseline_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_names = [name for name, _ in ranked[:top_n]]

    # ── Step 3 & 4: Tune + Guard ──────────────────────────────────────────────
    updated_models = dict(trained_models)  # shallow copy; we replace entries below
    tuning_summary: dict[str, Any] = {}

    for i, model_name in enumerate(top_names, start=1):
        if progress_callback:
            progress_callback(f"Tuning {model_name} ({i}/{len(top_names)})")

        param_grid = param_grids.get(model_name)
        baseline_score = baseline_scores.get(model_name, 0.0)

        if param_grid is None:
            # No grid defined for this model — skip tuning, still record it.
            tuning_summary[model_name] = {
                "was_tuned": False,
                "skip_reason": "No parameter grid defined for this model; tuning was skipped.",
                "baseline_cv_score": round(baseline_score, 6),
                "tuned_cv_score": None,
                "best_params": None,
                "was_improved": False,
                "reason": "Skipped because this algorithm does not have a tuned parameter grid configured.",
            }
            continue

        original_model = trained_models[model_name]

        try:
            search = RandomizedSearchCV(
                estimator=original_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=1,
                refit=True,
                error_score=0.0,
            )
            search.fit(X_train, y_train)

            tuned_score = float(search.best_score_)
            improved = tuned_score > baseline_score

            if improved:
                updated_models[model_name] = search.best_estimator_

            tuning_summary[model_name] = {
                "was_tuned": True,
                "baseline_cv_score": round(baseline_score, 6),
                "tuned_cv_score": round(tuned_score, 6),
                "best_params": search.best_params_,
                "was_improved": improved,
                "reason": (
                    "Tuned with RandomizedSearchCV because this model had a defined parameter grid and ranked in the top "
                    f"{top_n} by validation performance. "
                    + ("The tuned model was kept because its CV score improved from "
                       f"{round(baseline_score, 6)} to {round(tuned_score, 6)}." if improved else
                       "The tuned model was not kept because the CV score did not improve beyond the baseline.")
                ),
            }

        except Exception as exc:  # pragma: no cover – defensive
            tuning_summary[model_name] = {
                "was_tuned": False,
                "skip_reason": str(exc),
                "baseline_cv_score": round(baseline_score, 6),
                "tuned_cv_score": None,
                "best_params": None,
                "was_improved": False,
                "reason": f"Tuning was attempted but skipped because the search failed: {exc}",
            }

    # Record models that were not tuned (outside top N)
    for model_name in trained_models:
        if model_name not in tuning_summary:
            tuning_summary[model_name] = {
                "was_tuned": False,
                "skip_reason": f"Not in top {top_n} by CV score",
                "baseline_cv_score": round(baseline_scores[model_name], 6),
                "tuned_cv_score": None,
                "best_params": None,
                "was_improved": False,
                "reason": f"Not tuned because this model ranked outside the top {top_n} by cross-validation score and we only tuned the strongest candidates.",
            }

    if progress_callback:
        progress_callback("Hyperparameter tuning complete")

    return updated_models, tuning_summary
