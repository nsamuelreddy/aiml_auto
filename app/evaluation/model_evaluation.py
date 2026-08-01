import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_models(
    trained_models,
    predictions,
    X_test,
    y_test
):

    report = {}

    for name, model in trained_models.items():

        y_pred = predictions[name]

        report[name] = {

            "Accuracy": accuracy_score(y_test, y_pred),

            "Precision": precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

            "ROC-AUC": (
                roc_auc_score(
                    y_test,
                    model.predict_proba(X_test)[:, 1]
                )
                if hasattr(model, "predict_proba")
                else "Not Supported"
            ),

            "Confusion Matrix": confusion_matrix(
                y_test,
                y_pred
            ),

            "Classification Report": classification_report(
                y_test,
                y_pred,
                zero_division=0
            )

        }

    return report