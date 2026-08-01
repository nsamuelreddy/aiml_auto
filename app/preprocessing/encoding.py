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
    columns = binary_columns(df)
    for col in columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
        report[col] = "Label Encoding"
    return df, report, encoders
def one_hot_encoding(df):
    report={}
    columns=multi_categorical_columns(df)
    df=pd.get_dummies(df,columns=columns,dtype=int,drop_first=True)
    for col in columns:
        report[col]="One Hot Encoding"
    return df,report
def encoding_report(label_report,one_hot_report):
    report={}
    report.update(label_report)
    report.update(one_hot_report)
    return report