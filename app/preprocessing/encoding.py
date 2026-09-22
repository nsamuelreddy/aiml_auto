import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def binary_columns(df):
    categorical_cols= df.select_dtypes(exclude=np.number).columns
    binary=[]
    for col in categorical_cols:
        if df[col].nunique () == 2:
            binary.append(col)
    return binary
def multi_categorical_columns(df):
    categorical_cols=df.select_dtypes(exclude=np.number).columns
    multi=[]
    for col in categorical_cols:
        if 2 < df[col].nunique() <=10:
            multi.append(col)
    return multi
def label_encode(df):
    report = {}
    encoders = {}
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        # Encode binary columns and high-cardinality columns (> 10 unique values)
        if df[col].nunique() == 2 or df[col].nunique() > 10:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
            report[col] = "Label Encoding"
    return df, report, encoders
def one_hot_encoding(df):
    report={}
    columns=multi_categorical_columns(df)
    if columns:
        df=pd.get_dummies(df,columns=columns,dtype=int,drop_first=True)
        for col in columns:
            report[col]="One Hot Encoding"
    # Fallback safety: encode any remaining non-numerical columns
    remaining_non_num = df.select_dtypes(exclude=np.number).columns
    for col in remaining_non_num:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        report[col] = "Label Encoding"
    return df,report
def encoding_report(label_report,one_hot_report):
    report={}
    report.update(label_report)
    report.update(one_hot_report)
    return report