from sklearn.feature_selection import VarianceThreshold
import pandas as pd,numpy as np
def variance_threshold_selection(x, threshold=0.0):

    selector = VarianceThreshold(threshold=threshold)

    numerical_columns = x.select_dtypes(include=np.number).columns
    non_numerical_columns = x.columns.difference(numerical_columns)

    if len(numerical_columns) == 0:
        report = {
            col: "Removed (Non-numeric column)"
            for col in non_numerical_columns
        }
        return x.drop(columns=non_numerical_columns), report

    selected_data = selector.fit_transform(x[numerical_columns])

    selected_columns = numerical_columns[selector.get_support()]

    removed_columns = [
        col for col in numerical_columns
        if col not in selected_columns
    ]

    removed_columns.extend(non_numerical_columns.tolist())

    x = pd.DataFrame(
        selected_data,
        columns=selected_columns,
        index=x.index
    )

    report = {}

    for col in removed_columns:

        report[col] = f"Removed (Variance <= {threshold})"

    return x, report
def correlation_selection(x, threshold=0.90):

    numerical_columns = x.select_dtypes(include=np.number).columns

    correlation_matrix = x[numerical_columns].corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    removed_columns = []

    for column in upper_triangle.columns:

        if any(upper_triangle[column] > threshold):

            removed_columns.append(column)

    x = x.drop(columns=removed_columns)

    report = {}

    for col in removed_columns:

        report[col] = f"Removed (Correlation > {threshold})"

    return x, report
def feature_selection_report(
    variance_report,
    correlation_report
):

    report = {}

    report.update(variance_report)

    report.update(correlation_report)

    return report
