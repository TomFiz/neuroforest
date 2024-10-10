from io import BytesIO
from itertools import chain
from typing import Any, Iterable, List, Set, Tuple, TypeVar
import json

import PIL
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import ImageDraw, ImageOps, ImageFilter, ImageFont
from PIL.Image import Image
from PIL.ImageEnhance import Contrast
from adfluo.exceptions import ExtractionError
from adfluo import Extractor, F, Input, Feat, SampleProcessor, param, DatasetAggregator, Agg
from adfluo.dataset import SubsetLoader
from adfluo.processors import DSFeat
from adfluo.storage import StorageProtocol
from adfluo.types import SampleID, FeatureName
from scipy.spatial import ConvexHull

from neuroforest.dataset import DATA_FOLDER, SESSION_TYPES, Coordinates, TimeStampedCoordinates, NeuroForestSession, \
    dataloader

ZERO = 1e-10

### Custom Storage

class SummaryImgStorage(StorageProtocol):
    FOLDER = DATA_FOLDER / "explorations"

    def store(self, sample_id: SampleID, feat: FeatureName, value: Image):
        img_path = self.FOLDER / f"{sample_id}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        value.save(img_path, format="jpeg")


img_summary_storage = SummaryImgStorage()


class ExplorationImagesStorage(StorageProtocol):
    FOLDER = DATA_FOLDER / "explorations"

    def store(self, sample_id: SampleID, feat: FeatureName, value: Image):
        img_path = self.FOLDER / sample_id / f"{feat.split('/')[0]}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        value.save(img_path, format="png")


exploration_storage = ExplorationImagesStorage()


### Processors and functions

def calculate_asrs(questionnaire_answers: dict[str, Any]) -> int:
    asrs_score = 0
    asrs_columns_count = 0
    for column_name, value in questionnaire_answers.items():
        if column_name.startswith("asrs"):
            asrs_score += int(value)
            asrs_columns_count += 1
    assert asrs_columns_count == 18  # sanity check
    return asrs_score


def to_vect(coordinates: List[Coordinates]) -> np.ndarray:
    return np.array([coord.to_vect() for coord in coordinates])


def to_2d_vect(coordinates: np.ndarray) -> np.ndarray:
    return coordinates[:, 0:1]


def to_coords(timestamped_coords: List[TimeStampedCoordinates]) -> List[Coordinates]:
    return [ts["coord"] for ts in timestamped_coords]


def distance(a: Coordinates, b: Coordinates) -> float:
    return np.sqrt(((a.to_vect() - b.to_vect()) ** 2).sum())


def pairwise_distance(coords_a: List[Coordinates], coords_b: List[Coordinates]) -> np.ndarray:
    coords_a = to_vect((coords_a))
    coords_b = to_vect((coords_b))
    return np.sqrt(((coords_a[1:] - coords_b[:-1]) ** 2).sum(axis=1))


def consecutive_deltas(coordinates: List[Coordinates]) -> np.ndarray:
    coordinates = to_vect(coordinates)
    return np.sqrt(((coordinates[1:] - coordinates[:-1]) ** 2).sum(axis=1))


def timestamp_views(player_coords: List[TimeStampedCoordinates],
                    mushroom_in_view: List[Set[Coordinates]]) \
        -> List[Tuple[float, Set[Coordinates]]]:
    """Adds a timestamp to each mushroom"""
    pass


def time_to_index(t, total_time, total_samples_nb):
    return int(t * total_samples_nb / total_time)


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def first_mushroom_in_view(session: NeuroForestSession) -> List[Tuple[Coordinates, Coordinates]]:
    """Finds the first mushroom in view after a capture and associates it to the mushroom that
    is then captured"""
    assert len(session.player_coords) == len(session.mushroom_in_view)
    total_time = session.total_time
    total_samples = len(session.player_coords)
    captured_mushrooms = [session.player_coords[0]] + session.gathered_mushrooms
    mushroom_pairs = []
    for current_mushroom, next_mushroom in pairwise(captured_mushrooms):
        start_idx = time_to_index(current_mushroom["timestamp"], total_time, total_samples) + 1
        end_idx = time_to_index(next_mushroom["timestamp"], total_time, total_samples) - 1
        mushrooms_in_view = session.mushroom_in_view[start_idx:end_idx]
        for mushrooms in mushrooms_in_view:
            if mushrooms:
                distances = [distance(current_mushroom["coord"], m) for m in mushrooms]
                closest_m, _ = min(zip(mushrooms, distances), key=lambda x: x[1])
                mushroom_pairs.append((closest_m, next_mushroom["coord"]))
                break
    return mushroom_pairs


class CoordinatesTransform(SampleProcessor):
    zoom_factor: float = param(default=3.0)
    offset: Tuple[float, float] = param(default=(200, 200))

    def post_init(self):
        self.offset_array = np.array(self.offset)

    def transform(self, coords: List[Coordinates]) -> np.ndarray:
        return (to_vect(coords)[:, [0, 2]] + self.offset_array) * self.zoom_factor

    def process(self, session: NeuroForestSession) -> Tuple[np.ndarray, np.ndarray]:
        mushrooms_coords = self.transform(session.mushroom_coords)
        player_coords = self.transform([c["coord"] for c in session.player_coords])
        return mushrooms_coords, player_coords


def max_coords(coords: np.ndarray) -> Tuple[int, int]:
    max_coord_x = int(coords[:, 0].max())
    max_coord_z = int(coords[:, 1].max())
    return max_coord_x, max_coord_z


class ExplorationRenderer(SampleProcessor):

    @staticmethod
    def draw_cross(draw: ImageDraw.Draw, coord: tuple[int, int], time: int):
        offset = 10
        topleft = (coord[0] - offset, coord[1] - offset)
        topright = (coord[0] + offset, coord[1] - offset)
        bottomleft = (coord[0] - offset, coord[1] + offset)
        bottomright = (coord[0] + offset, coord[1] + offset)
        draw.line(topleft + bottomright, fill=(0, 0, 255), width=4)
        draw.line(topright + bottomleft, fill=(0, 0, 255), width=4)
        font = ImageFont.load_default(size=20)
        draw.text((coord[0] + 30, coord[1]), anchor="mm", fill="blue",
                  text=f"{time}mn", font=font, align="center")

    def process(self, data: Tuple[np.ndarray, np.ndarray]) -> Image:
        """Draws the mushrooms AND the trajectory. This is only for later visual exploration"""
        mushrooms_coords, player_coords = data
        max_coord_x = int(mushrooms_coords[:, 0].max())
        max_coord_z = int(mushrooms_coords[:, 1].max())

        with PIL.Image.new(mode="RGB", size=(max_coord_x, max_coord_z), color=(255, 255, 255)) as img:
            draw = ImageDraw.Draw(img)
            for m in mushrooms_coords:
                draw.ellipse((tuple((m - 10).astype(int)), tuple((m + 10).astype(int))),
                             fill=(255, 255, 255),
                             outline=(0, 0, 0))
            draw.line([tuple(p) for p in player_coords.astype(int)], fill=(255, 0, 0), width=8)
            for i, p in enumerate(player_coords):
                if i % (50 * 60) == 0:
                    self.draw_cross(draw, p.astype(int), i // (50 * 60))
            return img


class FractalTransform(SampleProcessor):
    max_log_power: int = param(default=14)

    @classmethod
    def make_window_average_filter(cls, size, delta_t):
        window = np.zeros(size)
        window[:delta_t] = 1 / delta_t
        return window

    @classmethod
    def convolve(cls, x, window, delta_t: int):
        return scipy.signal.fftconvolve(x, window, axes=0)[delta_t:x.shape[0]]

    @classmethod
    def compute_feature_deltat(cls, player_coords, delta_t):
        player_coords_deriv = player_coords[1:] - player_coords[:-1]
        player_dist = np.linalg.norm(player_coords_deriv, axis=1)

        window_average_filter = cls.make_window_average_filter(player_dist.shape[0], delta_t)
        player_dist_deltat = cls.convolve(player_dist, window_average_filter, delta_t)
        player_deriv_deltat = cls.convolve(player_coords_deriv, window_average_filter[:, None], delta_t)
        player_bird_distance_deltat = np.linalg.norm(player_deriv_deltat, axis=1)

        # removing 0 values indexes to prevent NaN
        zero_indexes = player_bird_distance_deltat != 0
        player_bird_distance_deltat = player_bird_distance_deltat[zero_indexes]
        player_dist_deltat = player_dist_deltat[zero_indexes]

        feature_deltat = np.mean(player_dist_deltat / player_bird_distance_deltat)
        return feature_deltat

    def process(self, player_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        deltats = [2 ** i for i in range(self.max_log_power)]
        features = [self.compute_feature_deltat(player_coords, delta_t) for delta_t in deltats]
        return np.array(deltats), np.array(features)


def plot_fractal_transform(transform_data: tuple[np.ndarray, np.ndarray]) -> Image:
    deltats, transform = transform_data
    plt.figure(figsize=(12, 3))
    plt.axhline(y=0, color='lightgray')
    deltats = np.array(deltats) / 50  # 50 samples/s
    plt.plot(deltats, transform, color='k')
    plt.scatter(deltats, transform, color='b', alpha=0.5)
    plt.xlabel("Fenêtre (en seconde)")
    plt.xscale('log')
    plt.ylim(0, 4)
    buffer = BytesIO()
    plt.savefig(buffer, format="jpg", bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return PIL.Image.open(buffer)


class Heatmap(DatasetAggregator):

    @staticmethod
    def get_trajectory_array(player_coords, img_size: tuple[int, int]) -> np.ndarray:
        with PIL.Image.new(mode="1", size=img_size) as im:
            draw = ImageDraw.Draw(im)
            draw.line([tuple(p) for p in player_coords.astype(int)], fill=1, width=2)
            return np.array(im)

    def aggregate(self, all_samples: List[Tuple[np.ndarray, np.ndarray]]) -> Image:
        all_mushroom_coords, all_players_coords = zip(*all_samples)
        max_coord_x = int(all_mushroom_coords[0][:, 0].max())
        max_coord_z = int(all_mushroom_coords[0][:, 1].max())
        img_size = (max_coord_x, max_coord_z)
        overlayed_trajectories = np.zeros(img_size, dtype=int).T + ZERO
        for player_data in all_players_coords:
            overlayed_trajectories += self.get_trajectory_array(player_data, img_size)
        log_scaled = np.log(overlayed_trajectories)
        log_scaled[log_scaled == log_scaled.min()] = 0
        normed = (log_scaled / log_scaled.max()) * 255
        img = PIL.Image.fromarray(normed.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
        img = Contrast(img).enhance(4)
        img = ImageOps.invert(img)  # convert to black-on-white
        return img


def overlay_trajectory(heatmap: Image, trajectory: Image) -> Image:
    output_img = heatmap.copy().convert("RGB")
    # binarizing the trajectory image to overlay all non-white parts
    alpha_array = (np.array(ImageOps.invert(trajectory)).sum(axis=2) != 0) * 255
    alpha_channel = PIL.Image.fromarray(alpha_array.astype(np.uint8), mode="L")
    output_img.paste(trajectory, mask=alpha_channel)
    return output_img


def upscale(img: Image, factor: float) -> Image:
    new_size = tuple((np.array(img.size) * factor).astype(int))
    return img.resize(new_size)


def image_vstack(*images: Image) -> Image:
    max_width_img = max(images, key=lambda img: img.width)
    resized_imgs = []
    for img in images:
        if img is max_width_img:
            resized_imgs.append(img)
            continue

        if img.width == max_width_img.width:
            resized_imgs.append(img)
            continue

        ratio = max_width_img.width / img.width
        resized_imgs.append(upscale(img, ratio))

    stacked_height = sum(img.height for img in resized_imgs)
    stacked_img = PIL.Image.new(mode="RGB",
                                size=(max_width_img.width, stacked_height))
    height = 0
    for img in resized_imgs:
        stacked_img.paste(img, (0, height))
        height += img.height
    return stacked_img


def image_hstack(*images: Image) -> Image:
    max_height_img = max(images, key=lambda img: img.height)
    resized_imgs = []
    for img in images:
        if img is max_height_img:
            resized_imgs.append(img)
            continue

        if img.height == max_height_img.height:
            resized_imgs.append(img)
            continue

        ratio = max_height_img.height / img.height
        resized_imgs.append(upscale(img, ratio))

    stacked_width = sum(img.width for img in resized_imgs)
    stacked_img = PIL.Image.new(mode="RGB",
                                size=(stacked_width, max_height_img.height))
    width = 0
    for img in resized_imgs:
        stacked_img.paste(img, (width, 0))
        width += img.width
    return stacked_img


def add_title(image: Image, title: str, height: int, font_size: int) -> Image:
    new_img_size = (image.width, image.height + height)
    with PIL.Image.new(mode=image.mode, size=new_img_size, color= "white") as new_img:
        new_img.paste(image, (0, height))
        font = ImageFont.load_default(size=font_size)
        draw = ImageDraw.Draw(new_img)
        draw.text((image.width / 2, height / 2), anchor="mm", fill="black",
                  text=title, font=font, align="center")
        return new_img


class SessionTitle(SampleProcessor):
    session: str = param()

    def process(self, image: Image) -> Image:
        return add_title(image, self.session, 50, 30)


class SampleTitle(SampleProcessor):

    def process(self, image: Image, asrs_score: int) -> Image:
        # use self.current sample to know which sample we're dealing with
        title = f"{self.current_sample.id} (ASRS: {asrs_score})"
        return add_title(image, title, 50, 40)


### Extractor
extractor = Extractor()

extractor.add_extraction(
    Input("questionnaire")
    >> F(calculate_asrs)
    >> Feat("asrs")
)

for session_type in SESSION_TYPES:
    # Nb of mushrooms gathered
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: len(session.gathered_mushrooms))
        >> Feat(f"{session_type}/performance")
    )

    # Player distance
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: distance(session.player_coords[0]["coord"], session.player_coords[-1]["coord"]))
        >> Feat(f"{session_type}/player_distance")
    )

    # Distance to gathered mushrooms
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: distance(session.gathered_mushrooms[0]["coord"], session.gathered_mushrooms[-1]["coord"]))
        >> Feat(f"{session_type}/gathered_mushrooms_distance")
    )

    # Player trajectory
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: consecutive_deltas(to_coords(session.player_coords)))
        >> F(lambda v: v.sum())
        >> Feat(f"{session_type}/player_trajectory")
    )

    # Gathered mushrooms trajectory
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: consecutive_deltas(to_coords(session.gathered_mushrooms)))
        >> F(lambda v: v.sum())
        >> Feat(f"{session_type}/gathered_mushrooms_trajectory")
    )

    # Gathered mushrooms deltas
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: consecutive_deltas(to_coords(session.gathered_mushrooms)))
        >> F(lambda v: list(v))
        >> Feat(f"{session_type}/gathered_mushrooms_deltas"),
        drop_on_save=True
    )

    # Convex hull of player trajectory
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: to_vect(to_coords(session.player_coords)))
        >> F(lambda v: ConvexHull(v))
        >> F(lambda hull: hull.volume)
        >> Feat(f"{session_type}/player_convex_hull")
    )

    # TODO: 2D projection of hull?

    # Convex hull of gathered mushrooms
    extractor.add_extraction(
        Input(session_type)
        >> F(lambda session: to_vect(to_coords(session.gathered_mushrooms)))
        >> F(lambda v: ConvexHull(v))
        >> F(lambda hull: hull.area)
        >> Feat(f"{session_type}/gathered_mushrooms_convex_hull")
    )

    # First mushroom in view
    extractor.add_extraction(
        Input(session_type)
        >> F(first_mushroom_in_view)
        >> F(lambda l: tuple(zip(*l)))
        >> F(lambda x: pairwise_distance(x[0], x[1]))
        >> F(lambda v: list(v))
        >> Feat(f"{session_type}/first_seen_mushroom_delta"),
        drop_on_save=True
    )

    # Capture ratio
    extractor.add_extraction(
        Input(session_type)
        >> (
                F(lambda session: set(m["coord"] for m in session.gathered_mushrooms))
                |
                F(lambda session: set(chain.from_iterable(session.mushroom_in_view)))
        )
        >> F(lambda captured, viewed: len(captured) / len(viewed))
        >> Feat(f"{session_type}/capture_ratio")
    )

    # Exploration images
    extractor.add_extraction(
        Input(session_type)
        >> CoordinatesTransform()
        >> ExplorationRenderer()
        >> Feat(f"{session_type}/exploration_img"),
    )

    # Fractal frequencies
    extractor.add_extraction(
        Input(session_type)
        >> CoordinatesTransform()
        >> F(lambda transformed_coords: transformed_coords[1])
        >> FractalTransform()
        >> Feat(f"{session_type}/fractal_frequencies")
    )

    # Heatmap
    extractor.add_extraction(
        Input(session_type)
        >> CoordinatesTransform()
        >> Heatmap()
        >> DSFeat(f"{session_type}/heatmap")
    )

    # Overlay heatmap and trajectory
    extractor.add_extraction(
        (DSFeat(f"{session_type}/heatmap") | Feat(f"{session_type}/exploration_img"))
        >> F(overlay_trajectory)
        >> Feat(f"{session_type}/overlay", storage=exploration_storage)
    )

    # Summary image
    extractor.add_extraction(
        (
                Feat(f"{session_type}/fractal_frequencies") >> F(plot_fractal_transform)
                |
                (DSFeat(f"{session_type}/heatmap") | Feat(f"{session_type}/exploration_img")) >> F(overlay_trajectory)
        )
        >> F(image_vstack)
        >> SessionTitle(session=session_type)
        >> Feat(f"{session_type}/summary_img")
    )

# Summary image
extractor.add_extraction(
    (
            (Feat("first/summary_img") | Feat("uniform/summary_img") | Feat("patchy/summary_img"))
            >> F(image_hstack)
            |
            Feat("asrs")
    )
    >> SampleTitle()
    >> Feat("sample_summary", storage=img_summary_storage)
)

if __name__ == '__main__':

    session_types = ["uniform","patchy"]
    features = ["asrs"]
    for session_type in session_types:
        features = features + [f"{session_type}/performance",
                    f"{session_type}/player_distance",
                    f"{session_type}/gathered_mushrooms_distance",
                    f"{session_type}/player_trajectory",
                    f"{session_type}/gathered_mushrooms_trajectory",
                    f"{session_type}/gathered_mushrooms_deltas",
                    f"{session_type}/player_convex_hull",
                    f"{session_type}/gathered_mushrooms_convex_hull",
                    f"{session_type}/first_seen_mushroom_delta",
                    f"{session_type}/capture_ratio",
                    f"{session_type}/fracal_frequencies"]
    
    extractor.extraction_DAG.prune_features(
        keep_only=features)
    subjects = [session.subject_name for session in dataloader]
    subjects.remove("Estelle")
    # print(f"Studying subjects {subjects}")

    try :
        loader = SubsetLoader(dataloader, subjects)
        extracted_features = extractor.extract_to_dict(loader, extraction_order="sample")
        with open(f'{DATA_FOLDER}/trajectory_features.json', 'w') as json_file:
            json.dump(extracted_features, json_file, indent=4)

    except ExtractionError as e :
        print(f"Error extracting data for subject {subjects} : \n{e}")