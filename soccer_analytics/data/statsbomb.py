import json
import requests

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from kloppy import statsbomb as kloppy_statsbomb
from kloppy.domain import create_event
from kloppy.domain.models.statsbomb.event import StatsBombEventFactory, StatsBombShotEvent, StatsBombPassEvent
from pathlib import Path
from typing import List, Optional

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
class CustomStatsBombShotEvent(StatsBombShotEvent):
    statsbomb_xg: float = None
    is_penalty: bool = False
    absolute_timestamp: datetime = None


@dataclass(repr=False)
class CustomStatsBombPassEvent(StatsBombPassEvent):
    absolute_timestamp: datetime = None


class CustomStatsBombEventFactory(StatsBombEventFactory):
    def __init__(self, game_start_timestamp: datetime):
        self.game_start_timestamp = game_start_timestamp

    def build_shot(self, **kwargs) -> StatsBombShotEvent:
        kwargs['statsbomb_xg'] = kwargs['raw_event']['shot']['statsbomb_xg']
        kwargs['is_penalty'] = kwargs["raw_event"]["shot"]["type"]["name"] == "Penalty"
        kwargs['absolute_timestamp'] = self.game_start_timestamp + timedelta(seconds=kwargs["timestamp"])
        return create_event(CustomStatsBombShotEvent, **kwargs)

    def build_pass(self, **kwargs) -> StatsBombPassEvent:
        kwargs['absolute_timestamp'] = self.game_start_timestamp + timedelta(seconds=kwargs["timestamp"])
        return create_event(CustomStatsBombPassEvent, **kwargs)


def get_events(season: Season, event_types: Optional[list[str]] = None):
    data_dir = Path(__file__).parent.parent.parent / "data"
    event_dir = data_dir / "statsbomb" / "events"
    lineup_dir = data_dir / "statsbomb" / "lineups"
    event_list = []
    for match in season.matches:
        event_file = event_dir / f"{match.match_id}.json"
        lineup_file = lineup_dir / f"{match.match_id}.json"
        if event_file.is_file() is False:
            event_data = requests.get(
                f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match.match_id}.json"
            )
            with open(event_file, "w") as f:
                json.dump(event_data.json(), f)
        if lineup_file.is_file() is False:
            lineup_data = requests.get(
                f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/lineups/{match.match_id}.json"
            )
        with open(lineup_file, "w") as f:
            json.dump(lineup_data.json(), f)

        try:
            events = kloppy_statsbomb.load(
                event_file, lineup_file,
                event_types=event_types, coordinates="statsbomb",
                event_factory=CustomStatsBombEventFactory(match.match_datetime)
            )
        except json.JSONDecodeError:
            print(f"Parse error for match_id {match.match_id}")

        event_list.extend(events)

    return event_list



if __name__ == "__main__":
    from pathlib import Path
    from kloppy.statsbomb import load
    data_path = Path(__file__).parent.parent.parent / "data" / "statsbomb"
    events = load(
        data_path / "events" / "7298.json" ,
        data_path / "lineups" / "7298.json",
        event_types=["shot"],
        coordinates="statsbomb",
        event_factory=CustomStatsBombEventFactory(datetime.strptime("2021-01-01_12:00:00", "%Y-%m-%d_%H:%M:%S"))
    )
    breakpoint()