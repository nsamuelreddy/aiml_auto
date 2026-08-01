import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
def standard_scale(x):
    numerical_columns = x.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    x[numerical_columns] = scaler.fit_transform(x[numerical_columns])
    report = {}
    for col in numerical_columns:
        report[col] = "Standard Scaling"
    return x, report, scaler

def min_max_scale(x):
    numerical_columns = x.select_dtypes(include=np.number).columns.tolist()
    scaler = MinMaxScaler()
    x[numerical_columns] = scaler.fit_transform(x[numerical_columns])
    report = {}
    for col in numerical_columns:
        report[col] = "Min-Max Scaling"
    return x, report, scaler