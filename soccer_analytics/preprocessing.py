import pandas as pd

from kloppy.domain.models import Point
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
            closest_defender_distance=get_closest_defender_distance,
            is_goal=lambda event: int(event.result.is_success),
            is_blocked=lambda event: int(event.result.name == "BLOCKED"),
            num_blockers=determine_blockers,
            timestamp=lambda event: event.timestamp,
            freeze_frame=lambda event: None if "freeze_frame" not in event.raw_event["shot"].keys() else event.raw_event["shot"]["freeze_frame"]
        ))
    return pd.concat(df_list)


def determine_blockers(event):
    shot_location = event.coordinates
    offset = 0.5
    if "freeze_frame" not in event.raw_event["shot"].keys() and event.raw_event["shot"]["type"]["name"] == "Penalty":
        return 0

    num_blockers = 0
    for entry in event.raw_event["shot"]["freeze_frame"]:
        if entry["teammate"] is True or entry["position"]["name"] == "Goalkeeper":
            continue

        player_position =  Point(*entry["location"])

        if shot_location.distance_to(player_position) > 5:
            continue

        if player_position.x < shot_location.x or player_position.x > 120:
            continue

        slope_upper = (shot_location.y + offset - 44) / (shot_location.x - 120)
        slope_lower = (shot_location.y - offset - 36) / (shot_location.x - 120)
        b_upper = (shot_location.y + offset) - slope_upper * shot_location.x
        b_lower = (shot_location.y - offset) - slope_lower * shot_location.x
        y_coord_upper = slope_upper * player_position.x + b_upper
        y_coord_lower = slope_lower * player_position.x + b_lower
        if player_position.y > y_coord_upper or player_position.y < y_coord_lower:
            continue

        num_blockers += 1


    return num_blockers

def get_closest_defender_distance(event):
    shot_location = event.coordinates
    closest_defender_distance = None
    if "freeze_frame" not in event.raw_event["shot"].keys() and event.raw_event["shot"]["type"]["name"] == "Penalty":
        return -999
    for entry in event.raw_event["shot"]["freeze_frame"]:
        if entry["teammate"] is False:
            player_location = Point(*entry["location"])
            distance = shot_location.distance_to(player_location)
            if closest_defender_distance is None or distance < closest_defender_distance:
                closest_defender_distance = distance
    return closest_defender_distance


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
