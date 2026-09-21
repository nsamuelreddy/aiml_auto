def clean_column_names(df):
    df.columns=(
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df
import numpy as np

def drop_irrelevant_columns(df, threshold=0.15):

    dropped_columns = []
    drop_reasons = {}

    # Always remove known ID columns
    id_columns = [
        "id",
        "passengerid",
        "loan_id",
        "customer_id",
        "employee_id",
        "student_id"
    ]

    for col in id_columns:
        if col in df.columns:
            dropped_columns.append(col)
            drop_reasons[col] = "Known ID column"

    # Get all categorical columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in categorical_cols:

        if col in dropped_columns:
            continue

        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > threshold:
            dropped_columns.append(col)
            drop_reasons[col] = f"High cardinality categorical column ({unique_ratio:.0%} unique values)"

    df = df.drop(columns=dropped_columns)

    return df, dropped_columns, drop_reasons
def remove_duplicate_rows(df):
    duplicates_removed=df.duplicated().sum()
    df=df.drop_duplicates()
    return df,duplicates_removed
def remove_empty_rows(df):
    df=df.dropna(how="all")
    return df
def remove_empty_columns(df):
    df=df.dropna(axis=1,how="all")
    return df
