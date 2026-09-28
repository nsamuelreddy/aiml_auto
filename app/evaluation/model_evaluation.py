import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize


def evaluate_models(
    trained_models,
    predictions,
    X_test,
    y_test
):

    report = {}

    for name, model in trained_models.items():

        y_pred = predictions[name]

        roc_auc = "Not Supported"
        pr_auc = "Not Supported"
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                unique_classes = sorted(list(set(y_test)))
                if len(unique_classes) == 2:
                    roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
                    pr_auc = float(average_precision_score(y_test, proba[:, 1]))
                elif len(unique_classes) > 2:
                    roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
                    y_bin = label_binarize(y_test, classes=unique_classes)
                    pr_auc = float(average_precision_score(y_bin, proba, average="weighted"))
            except Exception:
                roc_auc = "Not Supported"
                pr_auc = "Not Supported"

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

            "ROC-AUC": roc_auc,

            "PR-AUC": pr_auc,

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