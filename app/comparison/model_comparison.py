import pandas as pd


def compare_models(evaluation_report):

    comparison = []

    for model, metrics in evaluation_report.items():

        comparison.append({

            "Model": model,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "ROC-AUC": metrics["ROC-AUC"]

        })

    comparison = pd.DataFrame(comparison)

    comparison = comparison.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    return comparison


def top_models(comparison, top_n=3):

    return comparison.head(top_n)


def best_model(comparison):

    return comparison.iloc[0]