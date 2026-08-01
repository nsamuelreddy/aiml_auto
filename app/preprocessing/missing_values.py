import pandas as pd
import numpy as np


def numerical_missing_columns(df):
    """
    Returns numerical columns containing missing values.
    """

    numerical_cols = df.select_dtypes(include=np.number).columns

    missing_columns = []

    for col in numerical_cols:

        if df[col].isnull().sum() > 0:

            missing_columns.append(col)

    return missing_columns


def categorical_missing_columns(df):
    """
    Returns categorical columns containing missing values.
    """

    categorical_cols = df.select_dtypes(exclude=np.number).columns

    missing_columns=[]
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            missing_columns.append(col)
    return missing_columns


def detect_outliers(df, column):
    """
    Detects outliers using the IQR method.

    Returns True if outliers are present,
    otherwise False.
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    return len(outliers) > 0


def fill_numerical_missing(df):
    """
    Fills numerical missing values.

    Uses:
    Median -> if outliers exist
    Mean -> otherwise

    Returns:
    cleaned dataframe
    report dictionary
    """

    report = {}

    columns = numerical_missing_columns(df)

    for column in columns:

        if detect_outliers(df, column):

            median = df[column].median()

            df[column] = df[column].fillna(median)

            report[column] = "Median (Outliers Detected)"

        else:

            mean = df[column].mean()

            df[column] = df[column].fillna(mean)

            report[column] = "Mean (No Outliers)"

    return df, report


def fill_categorical_missing(df):
    """
    Fills categorical missing values using Mode.

    Returns:
    cleaned dataframe
    report dictionary
    """

    report = {}

    columns = categorical_missing_columns(df)

    for column in columns:

        mode = df[column].mode()[0]

        df[column] = df[column].fillna(mode)

        report[column] = "Mode"

    return df, report


def missing_value_report(numerical_report, categorical_report):
    """
    Combines numerical and categorical reports.
    """

    report = {}

    report.update(numerical_report)
    report.update(categorical_report)

    return report