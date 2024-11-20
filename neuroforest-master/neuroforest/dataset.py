import json
import pandas as pd
import os
import unicodedata
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, List, Set, Any, Union, Iterable
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/kupec/OneDrive/Desktop/neuroforest-main/neuroforest-master')
from adfluo import Sample, DatasetLoader

import numpy as np

DATA_FOLDER = Path(__file__).parent.parent.parent /"data_2022"
DATA_FOLDER_2024 = Path(__file__).parent.parent.parent /"data_2024"
SESSION_TYPES = {"first", "patchy", "uniform"}
SessionType = Literal["first", "patchy", "uniform"]

def remove_accents(input_str: str) -> str:
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

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

    def __init__(self,
                mushroom_coords: List[Coordinates],
                player_coords: List[TimeStampedCoordinates],
                mushroom_in_view: List[Set[Coordinates]],
                gathered_mushrooms: List[TimeStampedCoordinates],
                nb_mushrooms: int,
                total_time: int,):
        
        self.player_coords = player_coords
        # self.player_coords.sort(key=lambda e: e["timestamp"])
        self.gathered_mushrooms = gathered_mushrooms.sort(key=lambda e: e["timestamp"])
        self.mushroom_coords = mushroom_coords
        self.mushroom_in_view = mushroom_in_view
        self.gathered_mushrooms = gathered_mushrooms
        self.nb_mushrooms = nb_mushrooms
        self.total_time = total_time


class NeuroForestSample(Sample):

    def __init__(self, root_folder: Path,
                 subject_name: str,
                 questionnaire: dict[str, dict[str, Any]]):
        self.root_folder = root_folder
        self.subject_name = subject_name
        try :
            self.answers = questionnaire[remove_accents(self.subject_name.lower())]
        except KeyError:
            try :
                self.answers = questionnaire[remove_accents(self.subject_name[2:].lower())]
            except KeyError:
                self.answers = None

    @property
    def id(self) -> Union[str, int]:
        return self.subject_name

    def load_session(self, session_type: SessionType) -> NeuroForestSession:
        filepath = self.root_folder / f"{self.subject_name}_{session_type}0.json"
        if not os.path.exists(filepath):
            pass
        else :
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
                                        for e in json_content['ChampignonsRamassÃ©s']],
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
            try :
                self.questionnaire = {
                    remove_accents(row["Name"].lower()): row for row in dict_reader
                }
            except KeyError:
                df = pd.read_csv(questionnaire_data_path, sep = ";")
                self.questionnaire = {}
                for row in df.iterrows():
                    # convert row to dict
                    row_dict = dict(row[1])
                    if row[1].isna().sum() > 0 :
                        continue
                    self.questionnaire[remove_accents(row_dict["Name"].lower())] = row_dict
                

    @property
    def all_names(self):
        return {f.stem.split("_")[0] for f in self.folder.glob("*.json")}

    def __len__(self):
        return len(self.all_names)

    def __iter__(self) -> Iterable[Sample]:
        for name in self.all_names:
            yield NeuroForestSample(self.folder, name, self.questionnaire)


dataloader = NeuroForestLoader(DATA_FOLDER / "trajectories_processed",
                               DATA_FOLDER / "questionnaires/ASRS_Q.csv")
dataloader_2024 = NeuroForestLoader(DATA_FOLDER_2024 / "trajectories_processed",
                                 DATA_FOLDER_2024 /"Q_asrs_2024.csv")

print(f"Loading data from : {dataloader.folder}")
print(f"Loading data from : {dataloader_2024.folder}")
