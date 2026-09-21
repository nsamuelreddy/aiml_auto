import pandas as pd
import numpy as np
import os
import time


def load_dataset(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    start = time.time()

    if extension == ".csv":

        df = pd.read_csv(
            file_path,
            low_memory=True
        )

    elif extension in [".xls", ".xlsx"]:

        df = pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file format")

    load_time = round(time.time() - start, 2)

    print(f"\nDataset loaded in {load_time} seconds")

    memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"Memory Usage : {memory:.2f} MB")

    # Automatically sample huge datasets
    if len(df) > 100000:

        print(f"\nLarge dataset detected ({len(df)} rows)")
        print("Sampling 100000 rows for faster processing...")

        df = df.sample(
            n=100000,
            random_state=42
        ).reset_index(drop=True)

        print(f"Dataset after sampling : {len(df)} rows")

    return df


def dataset_shape(df):
    rows, columns = df.shape
    return rows, columns


def data_types(df):
    return df.dtypes


def missing_values(df):
    return df.isnull().sum()


def duplicate_rows(df):
    return df.duplicated().sum()


def memory_usage(df):
    return df.memory_usage(deep=True).sum() / 1024


def preview_dataset(df):
    return df.head()


def statistics(df):
    return df.describe(include="all")


def numerical_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()