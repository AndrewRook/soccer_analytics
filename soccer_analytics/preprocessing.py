import pandas as pd

from kloppy.domain.services.transformers.attribute import (
    BodyPartTransformer, AngleToGoalTransformer, DistanceToGoalTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin

def match_list_to_df(match_list):
    df_list = []
    for match in match_list:
        df_list.append(match.to_df(
            "is_penalty",
            "is_first_time",
            "statsbomb_xg",
            "technique",
            "coordinates*",
            AngleToGoalTransformer(),
            DistanceToGoalTransformer(),
            BodyPartTransformer(),
            position=lambda event: "Unknown" if event.player.position is None else event.player.position.name,
            is_goal=lambda event: int(event.result.is_success)
        ))
    return pd.concat(df_list)


class AngleNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, variable: str, new_variable: str):
        self.variable = variable
        self.new_variable = new_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        normalized_angle = (X[self.variable].abs() - 90).abs()
        X = X.drop(self.variable, axis=1)
        X[self.new_variable] = normalized_angle
        return X
