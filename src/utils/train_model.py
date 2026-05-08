from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline


def train_churn_model(X: DataFrame, y: Series, pipeline: Pipeline):
    return pipeline.fit(X, y)