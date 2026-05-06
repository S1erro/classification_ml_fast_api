from pandas import DataFrame

def is_dataframe_empty(df: DataFrame):
    if (df.empty):
        raise ValueError("No data")