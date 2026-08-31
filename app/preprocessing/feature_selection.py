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
def correlation_selection(x, threshold=0.95):
    numerical_columns = x.select_dtypes(include=np.number).columns
    if len(numerical_columns) < 2:
        return x, {}

    corr_matrix = x[numerical_columns].corr().abs()
    to_remove = set()
    cols = list(numerical_columns)

    # Keywords indicating critical domain features to prioritize keeping
    priority_keywords = ['loan_amount', 'loan', 'amount', 'price', 'income', 'score', 'salary', 'balance', 'total']

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            if col_a in to_remove or col_b in to_remove:
                continue
            if corr_matrix.loc[col_a, col_b] > threshold:
                score_a = sum(1 for kw in priority_keywords if kw in col_a.lower())
                score_b = sum(1 for kw in priority_keywords if kw in col_b.lower())

                if score_a > score_b:
                    to_remove.add(col_b)
                elif score_b > score_a:
                    to_remove.add(col_a)
                else:
                    # Drop the column with lower variance
                    if x[col_a].std() >= x[col_b].std():
                        to_remove.add(col_b)
                    else:
                        to_remove.add(col_a)

    removed_columns = list(to_remove)
    x = x.drop(columns=removed_columns)

    report = {col: f"Removed (Correlation > {threshold})" for col in removed_columns}
    return x, report
def feature_selection_report(
    variance_report,
    correlation_report
):

    report = {}

    report.update(variance_report)

    report.update(correlation_report)

    return report
