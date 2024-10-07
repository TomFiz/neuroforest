import json
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, List, Set, Any, Union, Iterable

from adfluo import Sample, DatasetLoader

import numpy as np

DATA_FOLDER = Path(__file__).parent.parent.parent /"data_2022"
SESSION_TYPES = {"first", "patchy", "uniform"}
SessionType = Literal["first", "patchy", "uniform"]


@dataclass(frozen=True)
class Coordinates:
    x: float
    y: float
    z: float

    def to_vect(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    @classmethod
    def from_vect(cls, vect: np.ndarray) -> 'Coordinates':
        return Coordinates(*vect)

    def translate(self, offset: 'Coordinates'):
        return Coordinates.from_vect(offset.to_vect() + self.to_vect())

    def transform(self, factor: float):
        return Coordinates.from_vect(self.to_vect() * factor)


class TimeStampedCoordinates(TypedDict):
    coord: Coordinates
    timestamp: float


@dataclass
class NeuroForestSession:
    mushroom_coords: List[Coordinates]
    player_coords: List[TimeStampedCoordinates]
    mushroom_in_view: List[Set[Coordinates]]
    gathered_mushrooms: List[TimeStampedCoordinates]
    nb_mushrooms: int
    total_time: int

    def __post_init__(self):
        self.player_coords.sort(key=lambda e: e["timestamp"])
        self.gathered_mushrooms.sort(key=lambda e: e["timestamp"])


class NeuroForestSample(Sample):

    def __init__(self, root_folder: Path,
                 subject_name: str,
                 questionnaire: dict[str, dict[str, Any]]):
        self.root_folder = root_folder
        self.subject_name = subject_name
        self.answers = questionnaire[self.subject_name.lower()]

    @property
    def id(self) -> Union[str, int]:
        return self.subject_name

    def load_session(self, session_type: SessionType) -> NeuroForestSession:
        filepath = self.root_folder / f"{self.subject_name}_{session_type}0.json"
        print(filepath)
        with filepath.open() as json_file:
            json_content = json.load(json_file)
            return NeuroForestSession(
                mushroom_coords=[Coordinates(**e)
                                 for e in json_content['PositionsChampis']],
                player_coords=[{"coord": Coordinates(**e[0]), "timestamp": e[1]}
                               for e in json_content['Positions']],
                mushroom_in_view=[{Coordinates(**e) for e in view}
                                  for view in json_content['Champignons dans champ de vision']],
                gathered_mushrooms=[{"coord": Coordinates(**e[0]), "timestamp": e[1]}
                                    for e in json_content['ChampignonsRamassés']],
                nb_mushrooms=json_content['Nb champignons'][0],
                total_time=json_content["Temps"][0]
            )

    def __getitem__(self, data_name: str) -> Any:
        if data_name in SESSION_TYPES:
            return self.load_session(data_name)
        elif data_name == "questionnaire":
            return self.answers


class NeuroForestLoader(DatasetLoader):

    def __init__(self, folder: Path, questionnaire_data_path: Path):
        self.folder = folder
        with questionnaire_data_path.open() as q_file:
            dict_reader = DictReader(q_file, delimiter="\t")
            self.questionnaire = {
                row["Name"].lower(): row for row in dict_reader
            }

    @property
    def all_names(self):
        return {f.stem.split("_")[0] for f in self.folder.glob("*.json")}

    def __len__(self):
        return len(self.all_names)

    def __iter__(self) -> Iterable[Sample]:
        for name in self.all_names:
            yield NeuroForestSample(self.folder, name, self.questionnaire)


dataloader = NeuroForestLoader(DATA_FOLDER / "trajectories",
                               DATA_FOLDER / "questionnaires/ASRS_Q.csv")
print(f"Loading data from : {dataloader.folder}")
