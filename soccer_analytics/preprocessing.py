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
            situation=lambda event: event.raw_event["shot"]["type"]["name"],
            closest_defender_distance=get_closest_defender_distance,
            is_goal=lambda event: int(event.result.is_success),
            is_blocked=lambda event: int(event.result.name == "BLOCKED"),
            timestamp=lambda event: event.timestamp,
            freeze_frame=lambda event: None if "freeze_frame" not in event.raw_event["shot"].keys() else event.raw_event["shot"]["freeze_frame"]
        ))
    return pd.concat(df_list)


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
    # Assuming the blocker is in a position to potentially impact the play,
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


def get_block_score(
        shot_location: Point,
        blocker_info: list[dict[str, Union[list, dict, str]]],
        block_score_function: Callable,
        goal_x=120.,
        goal_ys=(36, 44),
        goalie_strategy: str = "exclude",
        include_teammates: bool = False,
        overlap_strategy: str = "ignore",
        num_test_points=1000
):
    overlap_strategy = overlap_strategy.lower()
    if overlap_strategy not in ["ignore", "compound"]:
        raise ValueError("overlap_strategy must be one of 'ignore', 'compound'")

    goalie_strategy = goalie_strategy.lower()
    if goalie_strategy not in ["exclude", "include", "only"]:
        raise ValueError("goalie_strategy must be 'exclude', 'include', or 'only'")

    goal_y_values = np.linspace(*goal_ys, num_test_points)
    y_block_scores_per_player = np.zeros((num_test_points, len(blocker_info)))
    for i, blocker in enumerate(blocker_info):
        if blocker["position"]["name"] == "Goalkeeper":
            if goalie_strategy == "exclude":
                continue
        else:
            if goalie_strategy == "only":
                continue
        if include_teammates is False and blocker["teammate"] is True:
            continue
        blocker_distances = _map_goal_locations_to_distance_from_blocker(
            shot_location, Point(*blocker["location"]), goal_x, goal_y_values
        )
        y_block_scores_per_player[:, i] = block_score_function(
            blocker_distances, shot_location, blocker, goal_x
        )

    if overlap_strategy == "ignore":
        block_scores = np.max(
            y_block_scores_per_player,
            axis=1
        )
    else:
        # Treat each score like a probability. First get the "likelihood" of not blocking,
        # then do one minus that:
        no_block_scores_per_player = 1 - y_block_scores_per_player
        no_block_scores = np.multiply.reduce(
            no_block_scores_per_player,
            axis=1
        )
        block_scores = 1 - no_block_scores
    return block_scores.mean()


def uniform_block_score(block_score=0.9, max_distance=0.5):
    def inner(blocker_distances, shot_location, blocker, goal_x):
        score = np.where(blocker_distances < max_distance, block_score, 0)
        return score
    return inner


class BlockScore(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            freeze_frame_column: str,
            x_coordinate_column: str,
            y_coordinate_column: str,
            block_score_function: Callable,
            block_score_column_name: str = "block_score",
            **get_block_score_kwargs):
        self.freeze_frame_column = freeze_frame_column
        self.x_coordinate_column = x_coordinate_column
        self.y_coordinate_column = y_coordinate_column
        self.block_score_function = block_score_function
        self.block_score_column_name = block_score_column_name
        self.get_block_score_kwargs = get_block_score_kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        block_score = X.apply(
            lambda row: 0 if row[self.freeze_frame_column] is None else get_block_score(
                Point(row[self.x_coordinate_column], row[self.y_coordinate_column]),
                row[self.freeze_frame_column],
                self.block_score_function,
                **self.get_block_score_kwargs
            ),
            axis=1
        )
        X[self.block_score_column_name] = block_score
        return X


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
