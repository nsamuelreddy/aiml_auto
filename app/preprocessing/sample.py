import pandas as pd
file_path = "datasets/titanic.csv"

df = pd.read_csv(file_path)
print(df.isnull().sum())
