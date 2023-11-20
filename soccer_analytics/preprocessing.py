import math
import numpy as np
import pandas as pd

from kloppy.domain.models import Point
from kloppy.domain.services.transformers.attribute import (
    BodyPartTransformer, AngleToGoalTransformer, DistanceToGoalTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Callable, Union


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
            #num_blockers=determine_blockers,
            timestamp=lambda event: event.timestamp,
            freeze_frame=lambda event: None if "freeze_frame" not in event.raw_event["shot"].keys() else event.raw_event["shot"]["freeze_frame"]
        ))
    return pd.concat(df_list)


# def determine_blockers(event):
#     shot_location = event.coordinates
#     offset = 0.5
#     if "freeze_frame" not in event.raw_event["shot"].keys() and event.raw_event["shot"]["type"]["name"] == "Penalty":
#         return 0
#
#     num_blockers = 0
#     for entry in event.raw_event["shot"]["freeze_frame"]:
#         if entry["teammate"] is True or entry["position"]["name"] == "Goalkeeper":
#             continue
#
#         player_position =  Point(*entry["location"])
#
#         if shot_location.distance_to(player_position) > 5:
#             continue
#
#         if player_position.x < shot_location.x or player_position.x > 120:
#             continue
#
#         slope_upper = (shot_location.y + offset - 44) / (shot_location.x - 120)
#         slope_lower = (shot_location.y - offset - 36) / (shot_location.x - 120)
#         b_upper = (shot_location.y + offset) - slope_upper * shot_location.x
#         b_lower = (shot_location.y - offset) - slope_lower * shot_location.x
#         y_coord_upper = slope_upper * player_position.x + b_upper
#         y_coord_lower = slope_lower * player_position.x + b_lower
#         if player_position.y > y_coord_upper or player_position.y < y_coord_lower:
#             continue
#
#         num_blockers += 1
#
#
#     return num_blockers

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


def _map_goal_locations_to_distance_from_blocker(
        shot_location: Point,
        blocker_location: Point,
        goal_x: float,
        goal_y_values: np.array
):
    """
    Given an array of points along a goalmouth (or the entire end-line, really),
    Identify how close to the line drawn between the shot location and the blocker
    location is to the line between the shot location and each point on the goalmouth.

    The minimum distance occurs at the point perpendicular to the line drawn between the shot
    location and the blocker location, so this problem collapses to identifying that distance.
    """
    # First, if the blocker is not between the shot and the goal, return a null value:
    if blocker_location.x <= shot_location.x:
        return np.full_like(goal_y_values, np.nan)
    # Assuming the blocker is in a position to potentiall impact the play,
    # get the angle between the shot and the blocker compared to the horizontal axis:
    shot_blocker_angle = math.atan2(
        blocker_location.y - shot_location.y,
        blocker_location.x - shot_location.x
    )
    # The distance between the shot and the blocker
    shot_blocker_distance = math.sqrt(
        (blocker_location.y - shot_location.y) ** 2
        + (blocker_location.x - shot_location.x) ** 2
    )
    # Get the differences in y and x values between the shot location and the goalline:
    x_diff = goal_x - shot_location.x
    y_diff = goal_y_values - shot_location.y

    # The min distance is the tangent between the incremental angle of the y_value on the
    # goalline above the shot/blocker angle, multiplied by the distance between the shot and
    # the blocker:
    min_distances = shot_blocker_distance * np.tan(np.arctan2(y_diff, x_diff) - shot_blocker_angle)
    return np.abs(min_distances)


def get_block_likelihood(
        shot_location: Point,
        blocker_info: list[dict[str, Union[list, dict, str]]],
        block_function: Callable,
        goal_x=120.,
        goal_ys=(36, 44),
        num_test_points=1000
):
    goal_y_values = np.linspace(*goal_ys, num_test_points)
    y_block_likelihoods_per_player = np.zeros((num_test_points, len(blocker_info)))
    for i, blocker in enumerate(blocker_info):
        blocker_distances = _map_goal_locations_to_distance_from_blocker(
            shot_location, Point(*blocker["location"]), goal_x, goal_y_values
        )
        y_block_likelihoods_per_player[:, i] = block_function(
            blocker_distances, shot_location, blocker, goal_x
        )

    no_block_likelihoods_per_player = 1 - y_block_likelihoods_per_player
    no_block_likelihoods = np.multiply.reduce(
        no_block_likelihoods_per_player,
        axis=1
    )
    return 1 - no_block_likelihoods.mean()


def uniform_block_function(block_likelihood=0.9, max_distance=0.5):
    def inner(blocker_distances, shot_location, blocker, goal_x):
        likelihood = np.where(blocker_distances < max_distance, block_likelihood, 0)
        return likelihood
    return inner


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
