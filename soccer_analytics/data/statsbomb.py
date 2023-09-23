import requests

from dataclasses import dataclass, field
from datetime import datetime
from kloppy.domain import EventFactory, create_event, ShotEvent
from typing import List

URL_PREFIX = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
MATCHES_PREFIX = f"{URL_PREFIX}matches/"
COMPETITIONS_URL = f"{URL_PREFIX}competitions.json"

@dataclass
class Match:
    match_id: int
    match_datetime: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int


@dataclass
class Season:
    season_id: int
    name: str
    has_360: bool
    matches: List[Match] = field(default_factory=list)

@dataclass
class Competition:
    competition_id: int
    country_name: str
    name: str
    gender: str
    youth: bool
    international: bool
    seasons: List[Season] = field(default_factory=list)

def _get_metadata():
    competitions_json = requests.get(COMPETITIONS_URL).json()
    competitions = {}
    for competition in competitions_json:
        competition_obj = Competition(
            competition["competition_id"],
            competition["country_name"],
            competition["competition_name"],
            competition["competition_gender"],
            competition["competition_youth"],
            competition["competition_international"]
        )
        match_data = requests.get(
            f"{MATCHES_PREFIX}{competition['competition_id']}/{competition['season_id']}.json"
        ).json()
        matches = []
        for match in match_data:
            if match["kick_off"] is not None:
                match_datetime = datetime.strptime(match["match_date"] + "_" + match["kick_off"], "%Y-%m-%d_%H:%M:%S.%f")
            else:
                match_datetime = datetime.strptime(match["match_date"], "%Y-%m-%d")
            matches.append(Match(
                match["match_id"],
                match_datetime,
                match["home_team"]["home_team_name"],
                match["away_team"]["away_team_name"],
                match["home_score"],
                match["away_score"]
            ))
        season = Season(
            competition["season_id"],
            competition["season_name"],
            competition["match_available_360"] is not None,
            matches
        )
        if competition["competition_id"] in competitions:
            competitions[competition["competition_id"]].seasons.append(season)
        else:
            competition_obj.seasons.append(season)
            competitions[competition["competition_id"]] = competition_obj

    competitions = [value for _, value in competitions.items()]
    return competitions


class Metadata:
    def __init__(self):
        self.metadata = None

    def __call__(self, use_cache=True):
        if use_cache is False or self.metadata is None:
            self.metadata = _get_metadata()
        return self.metadata


get_metadata = Metadata()


@dataclass(repr=False)
class StatsBombShotEvent(ShotEvent):
    statsbomb_xg: float = None
    is_penalty: bool = False


class StatsBombEventFactory(EventFactory):
    def build_shot(self, **kwargs) -> ShotEvent:
        kwargs['statsbomb_xg'] = kwargs['raw_event']['shot']['statsbomb_xg']
        kwargs['is_penalty'] = kwargs["raw_event"]["shot"]["type"]["name"] == "Penalty"
        return create_event(StatsBombShotEvent, **kwargs)


STATSBOMB_EVENT_FACTORY = StatsBombEventFactory()


if __name__ == "__main__":
    metadata = get_metadata()
    breakpoint()