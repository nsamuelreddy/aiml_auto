import pandas as pd
def compare_regression_models(evaluation_report):
    comparison= pd.DataFrame(evaluation_report).T
    comparison=comparison.sort_values(
        by=["R2","RMSE"],
        ascending=[False,True]
    )
    return comparison.reset_index().rename(
        columns={"index":"Model"}
    )
def top_models(comparison,top_n=3):
    return comparison.head(top_n)
def best_model(comparison):
    return comparison.iloc[0].to_dict()
