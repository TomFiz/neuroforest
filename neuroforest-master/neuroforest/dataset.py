import json
from re import X
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import unicodedata
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, List, Set, Any, Union, Iterable
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Iterable
from PIL import Image
from scipy.spatial import ConvexHull


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
    
    def get_mushroom_convex_hull(self):
        """
        Calcule l'enveloppe convexe des positions de champignons.

        Returns:
            hull (ConvexHull): L'objet ConvexHull représentant l'enveloppe convexe.
            hull_points (ndarray): Les points de l'enveloppe convexe (coordonnées des sommets).
        """
        # Récupérer les coordonnées des champignons
        points = np.array([(c.x, c.y) for c in self.mushroom_coords])
        
        if len(points) < 3:
            raise ValueError("L'enveloppe convexe nécessite au moins trois points.")

        # Calculer l'enveloppe convexe
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return hull, hull_points

    def plot_convex_hull(self):
        """
        Trace les positions des champignons et l'enveloppe convexe.
        """
        hull, hull_points = self.get_mushroom_convex_hull()
        points = np.array([(c.x, c.y) for c in self.mushroom_coords])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], label="Champignons", alpha=0.6)
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label="Enveloppe convexe")
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.2, label="Zone Convexe")
        plt.legend()
        plt.title("Enveloppe Convexe des Champignons")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()



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
    
    @property
    def num_id(self): #ID numérique consistent avec le questionnaire
        return self.answers["sujet"] if self.answers is not None else None


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


class ImageLoader(Dataset):
    
    def __init__(self, root_folder: Path, questionnaire_data_path: Path):
        self.root_folder = root_folder
        if not self.root_folder.exists():
            raise FileNotFoundError(f"Le dossier {self.root_folder} n'existe pas.")
        self.image_files = list(self.root_folder.glob("*.png"))
        
        ##Pour les questionnaires
        with questionnaire_data_path.open() as q_file:
            dict_reader = DictReader(q_file, delimiter="\t")
            try:
                self.questionnaire = {
                    remove_accents(row["Name"].lower()): row for row in dict_reader
                }
            except KeyError:
                df = pd.read_csv(questionnaire_data_path, sep=";")
                self.questionnaire = {}
                for row in df.iterrows():
                    # convert row to dict
                    row_dict = dict(row[1])
                    if row[1].isna().sum() > 0:
                        continue
                    self.questionnaire[remove_accents(row_dict["Name"].lower())] = row_dict

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.image_categories = self.assign_categories()

    def parse_filename(self, filename: str) -> Dict[str, Optional[str]]:
        name, _ = filename.split(".")  # Supprimer l'extension
        parts = name.split("_")

        parsed_data = {
            "name": remove_accents(parts[0].lower()),  # Normaliser le nom
            "session_type": parts[1],
            "no_mushroom": "nomushroom" in parts,  # Boolean
            "no_timestamp": "notimestamp" in parts,  # Boolean
        }
        return parsed_data
    
    def get_category(self, idx: int) -> str:
        image_file = self.image_files[idx]
        return self.image_categories[image_file]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, str]]:
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        
        info = self.parse_filename(image_file.name)
        name = info["name"]
        label = self.questionnaire.get(name, {})
        
        return image, label
    
    def get_indices_by_category(self, category: str) -> List[int]:
        return [
            idx for idx, file in enumerate(self.image_files)
            if self.image_categories[file] == category
        ]


    def assign_categories(self) -> Dict[Path, str]:
        categories = {}
        for image_file in self.image_files:
            info = self.parse_filename(image_file.name)
            if not info["no_mushroom"] and not info["no_timestamp"]:
                categories[image_file] = "with_mushroom_with_timestamp"
            elif not info["no_mushroom"] and info["no_timestamp"]:
                categories[image_file] = "with_mushroom_no_timestamp"
            elif info["no_mushroom"] and not info["no_timestamp"]:
                categories[image_file] = "no_mushroom_with_timestamp"
            elif info["no_mushroom"] and info["no_timestamp"]:
                categories[image_file] = "no_mushroom_no_timestamp"
        return categories

    def get_image(self, name: str, session_type: str, no_mushroom: bool = False, no_timestamp: bool = False) -> Optional[Path]:
        for image_file in self.image_files:
            info = self.parse_filename(image_file.name)
            if (
                info["name"] == name and
                info["session_type"] == session_type and
                info["no_mushroom"] == no_mushroom and
                info["no_timestamp"] == no_timestamp
            ):
                return image_file
        print(f"Aucune image trouvée pour {name} - {session_type} avec no_mushroom={no_mushroom}, no_timestamp={no_timestamp}")
        return None

    def __iter__(self) -> Iterable[Path]:
        for image_file in self.image_files:
            yield image_file

# image_loader = ImageLoader(Path(__file__).parent.parent.parent / "images_2022", DATA_FOLDER / "questionnaires/ASRS_Q.csv" )
dataloader = NeuroForestLoader(DATA_FOLDER / "trajectories_processed",
                            DATA_FOLDER / "questionnaires/ASRS_Q.csv")
dataloader_2024 = NeuroForestLoader(DATA_FOLDER_2024 / "trajectories_processed",
                            DATA_FOLDER_2024 /"Q_asrs_2024_complet.csv")

print(f"Loading data from : {dataloader.folder}")
print(f"Loading data from : {dataloader_2024.folder}")