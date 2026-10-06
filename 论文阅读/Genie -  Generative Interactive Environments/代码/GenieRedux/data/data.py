import json
import logging
import math
import random
from multiprocessing import Lock, cpu_count
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from packaging.version import Version
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision import transforms as T
from tqdm import tqdm

from data_generation.data.data import (
    DEFAULT_VERSION,
    DatasetFileStructure,
    DatasetFileStructureInstance,
)
from tools.logger import getLogger

log = getLogger("data", name_color="blue")

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# image related helpers functions and dataset


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, exts=["jpg", "jpeg", "png"]):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f"{folder}").glob(f"**/*.{ext}")]

        print(f"{len(self.paths)} training samples found at {folder}")

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(image_size),
                T.RandomHorizontalFlip(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif


CHANNELS_TO_MODE = {1: "L", 3: "RGB", 4: "RGBA"}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f"channels {channels} invalid"
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_pil_images(tensor, only_first_image=True):
    tensor = torch.clamp(tensor, min=0, max=1)  # clipping underflow and overflow

    if only_first_image:
        return T.ToPILImage()(tensor.unbind(dim=1)[0])

    # convert all images to PIL and concatenate them to a single PIL image
    return T.ToPILImage()(torch.cat([t for t in tensor.unbind(dim=1)], dim=2))


def video_tensor_to_gif(
    tensor, path, duration=120, loop=0, optimize=True, actions=None
):
    tensor = torch.clamp(tensor, min=0, max=1)  # clipping underflow and overflow
    images = map(T.ToPILImage(), tensor.unbind(dim=1))

    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        loop=loop,
        optimize=optimize,
        duration=80,
    )
    return images


# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def tqdm_function_decorator(total, *args, **kwargs):
    class PbarFunctionDecorator(object):
        def __init__(self, func):
            self.func = func
            self.pbar = tqdm(total=total, *args, **kwargs)

        def __call__(self, *args, **kwargs):
            tmp = self.func(*args, **kwargs)
            self.pbar.update()
            return tmp

    return PbarFunctionDecorator


class DatasetFileStructureSessionLibrary:
    """A collection of all subject file structures in an environment dataset."""

    def __init__(self, root: str | Path, version=DEFAULT_VERSION, **kwargs) -> None:
        """Build a collection of all subject file structures in the Dreyeve dataset."""
        root = Path(root)
        self.fs = DatasetFileStructure(str(root), version=version, **kwargs)
        self.version = self.fs.version
        self.instance_ids = self.fs.get_instance_ids()
        self.sessions = {
            i: DatasetFileStructureInstance(root, i, self.fs.version, **kwargs)
            for i in self.instance_ids  # type: ignore
        }

    def __getitem__(self, key: int) -> DatasetFileStructureInstance:
        """Get a subject file structure by its ID."""
        return self.sessions[key]

    def __iter__(self):
        """Iterate over the subjects."""
        return iter(self.sessions.values())

    def __len__(self):
        """Return the number of subjects."""
        return len(self.sessions)


class DatasetOutputFormat:
    PVG = "pvg"
    IVG = "ivg"
    LAM = "lam"
    VICREG = "vicreg"
    GENERAL = "general"


class CombinedEnvironmentDataset(Dataset):
    def __init__(self, datasets: List[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets) * max([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        dataset_id = idx % len(self.datasets)
        dataset = self.datasets[dataset_id]
        sample_id = math.floor(idx / len(self.datasets)) % len(dataset)

        return dataset[sample_id]


class MultiEnvironmentDataset(Dataset):
    def __init__(
        self,
        root_dpath,
        seq_length_input: int = 5,
        seq_step: int = 1,
        enable_test: bool = False,
        split: str = "train",
        split_type: str = "frame",
        clip_length: int = 0,
        max_size: int | None = None,
        format: str = DatasetOutputFormat.GENERAL,
        transform=None,
        img_ext="jpg",
        instance_filter: dict | None = None,
        enable_cache: bool = True,
        cache_dpath: str = "cache",
        occlusion_mask: np.ndarray | None = None,
        source_dataset: Optional["MultiEnvironmentDataset"] = None,
        n_workers: int = 46,
        n_envs: int = 0,
        whitelist: list[str] = None,
        n_samples: int = -1,
        n_actions: int = -1,
    ) -> None:
        """
        Initializes the Data class.

        Args:
            root_dpath (str): The root directory path.
            seq_length_input (int, optional): The input sequence length. Defaults to 5.
            seq_step (int, optional): The sequence step. Defaults to 1.
            enable_test (bool, optional): Flag to enable test mode. Defaults to False.
            split (str, optional): The dataset split. Defaults to "train".
            split_type (str, optional): The split type. One of "instance" or "frame". Defaults to "frame".
            clip_length (int, optional): The clip length. If set to 0, the clip length is set to 1e12. Defaults to 0.
            max_size (int, optional): The maximum size of the dataset. Defaults to None.
            format (str, optional): The dataset format. Defaults to DatasetOutputFormat.GENERAL.
            transform (optional): The data transformation function. Defaults to None.
            img_ext (str, optional): The image file extension. Defaults to "jpg".
            instance_filter (dict, optional): A dict of acceptable session properties to use. Defaults to None.
            enable_cache (bool, optional): Flag to enable caching. Defaults to True.
            cache_dpath (str, optional): The cache directory path. Defaults to "cache".
            occlusion_mask (np.ndarray, optional): The occlusion mask to use. Defaults to None.
            n_workers (int, optional): The number of workers to use. Defaults to 46.
            n_envs (int, optional): The number of environments to use. Negative value denotes using the last n environments. The value 0 denotes all environments. Defaults to 0.
            whitelist (list[str], optional): A list of environment names to use. Defaults to None.
            n_samples (int, optional): The number of samples per environment to use. Negative value denotes using all samples. Defaults to -1.
        """
        use_last_n = n_envs < 0
        n_envs = abs(n_envs)
        assert n_envs == 0 or whitelist is None or len(whitelist) <= n_envs, (
            "Whitelist length should be less than n_envs"
        )

        self.log = getLogger("MultiEnvironmentDataset", name_color="blue")

        paths = sorted(
            [path for path in Path(root_dpath).iterdir() if path.is_dir()],
            key=lambda x: x.name,
        )  # [:5]

        paths_selected = []
        if whitelist is not None:
            paths_whitelist = [
                path for path in paths if path.name.split("_")[1] in whitelist
            ]
            paths_selected.extend(paths_whitelist)

        if whitelist is None:
            paths_no_whitelist = paths
            whitelist_len = 0
        else:
            paths_no_whitelist = [
                path for path in paths if path.name.split("_")[1] not in whitelist
            ]
            whitelist_len = len(whitelist)

        if n_envs > 0:
            if use_last_n:
                paths_selected.extend(paths_no_whitelist[-n_envs + whitelist_len :])
            else:
                paths_selected.extend(paths_no_whitelist[: n_envs - whitelist_len])
        else:
            paths_selected.extend(paths_no_whitelist)

        if n_envs > 0 or whitelist is not None:
            paths = paths_selected

        # paths = paths[-200:]

        # Optional: reuse per-environment metadata from an existing MultiEnvironmentDataset
        self.env_dataset_lookup = None
        if source_dataset is not None:
            self.env_dataset_lookup = {}
            for ds in source_dataset.datasets:
                base = ds.dataset if isinstance(ds, Subset) else ds
                env_root = str(base.fsl.fs.root_dpath)
                self.env_dataset_lookup[env_root] = base

        def create_environment_dataset(path, id):
            self.log.d(f"Creating dataset {id} from {path}")
            try:
                source_dataset = None
                if self.env_dataset_lookup is not None:
                    source_dataset = self.env_dataset_lookup[str(path)]
                dataset = EnvironmentDataset(
                    root_dpath=path,
                    seq_length_input=seq_length_input,
                    seq_step=seq_step,
                    enable_test=enable_test,
                    split=split,
                    split_type=split_type,
                    clip_length=clip_length,
                    max_size=max_size,
                    format=format,
                    transform=transform,
                    img_ext=img_ext,
                    instance_filter=instance_filter,
                    enable_cache=enable_cache,
                    cache_dpath=cache_dpath,
                    occlusion_mask=occlusion_mask,
                    source_dataset=source_dataset,
                )
            except KeyError as e:
                self.log.e(f"KeyError loading dataset from {path}: {e}")
                dataset = None
            except ValueError as e:
                self.log.e(f"ValueError loading dataset from {path}: {e}")
                dataset = None
            if n_actions > 0:
                dataset.n_actions = n_actions

            return dataset

        with ThreadPool(n_workers) as pool:
            self.datasets = pool.starmap(
                create_environment_dataset, [(path, i) for i, path in enumerate(paths)]
            )
        self.datasets = [dataset for dataset in self.datasets if dataset is not None]

        if len(self.datasets) == 0:
            raise ValueError("Empty dataset!")

        if n_samples > 0:
            self.datasets = [
                Subset(dataset, range(0, min(len(dataset), n_samples)))
                for dataset in self.datasets
            ]

        self.log.i(f"Loaded {len(self.datasets)} datasets")
        self.size = sum([len(dataset) for dataset in self.datasets])

        self.cummulative_size = np.cumsum([len(dataset) for dataset in self.datasets])
        self.n_datasets = len(self.datasets)
        # self.current_dataset_id = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # find out in which range in cummulative size the idx falls in
        dataset_id = np.argmax(self.cummulative_size > idx)
        new_idx = idx - self.cummulative_size[dataset_id - 1] if dataset_id > 0 else idx
        # breakpoint()
        try:
            item = self.datasets[dataset_id][new_idx]
        except IndexError as e:
            self.log.e(
                "IndexError: ",
                new_idx,
                idx,
                len(self.datasets[dataset_id]),
                self.cummulative_size,
                len(self),
            )
            raise e
        # self.current_dataset_id = (self.current_dataset_id + 1) % self.n_datasets
        return item


class EnvironmentDataset(Dataset):
    __DATASET_SPLIT = {
        "train": [0, 0.9],
        "validation": [0.9, 0.99],
        "test": [0.99, 1],
        "all": [0, 1],
    }

    def __init__(
        self,
        root_dpath,
        seq_length_input: int = 5,
        seq_step: int = 1,
        enable_test: bool = False,
        split: str = "train",
        split_type: str = "frame",
        clip_length: int = 0,
        max_size: int | None = None,
        format: str = DatasetOutputFormat.GENERAL,
        transform=None,
        img_ext="jpg",
        instance_filter: dict | None = None,
        enable_cache: bool = True,
        cache_dpath: str = "cache",
        occlusion_mask: np.ndarray | None = None,
        source_dataset: Optional["EnvironmentDataset"] | None = None,
    ) -> None:
        """
        Initializes the Data class.

        Args:
            root_dpath (str): The root directory path.
            seq_length_input (int, optional): The input sequence length. Defaults to 5.
            seq_step (int, optional): The sequence step. Defaults to 1.
            enable_test (bool, optional): Flag to enable test mode. Defaults to False.
            split (str, optional): The dataset split. Defaults to "train".
            split_type (str, optional): The split type. One of "instance" or "frame". Defaults to "frame".
            clip_length (int, optional): The clip length. If set to 0, the clip length is set to 1e12. Defaults to 0.
            format (str, optional): The dataset format. Defaults to DatasetOutputFormat.GENERAL.
            transform (optional): The data transformation function. Defaults to None.
            instance_filter (optional): A dict of acceptable session properties to use. Defaults to None.
        """

        self.seq_length_input = seq_length_input
        self.seq_length_current = seq_length_input
        self.seq_step = seq_step
        self.enable_test = enable_test
        self.format = format
        self.transform = transform
        self.session_filter = instance_filter
        self.occlusion_mask = occlusion_mask
        self.split = split
        self.split_type = split_type
        self.clip_length = clip_length

        # self.jitter_transform = torchvision.transforms.ColorJitter(brightness= 0.05, contrast= 0, saturation = 0.2, hue = 0.5)

        self.log = getLogger("data:EnvironmentDataset", name_color="blue")

        self.log.i(f"Creating dataset from {root_dpath}")
        if self.occlusion_mask is not None:
            self.log.d("Using occlusion mask!")

        self.info = EnvironmentDataset.__read_info(
            DatasetFileStructure(root_dpath).info_fpath
        )
        self.name = (
            self.info["name"]
            if self.info["name"] != "default"
            else Path(root_dpath).name
        )
        self.actions_shape = self.info["action_space"]

        # Determine output mode from generator config (frame vs video). Default to frames.
        self.output_mode = str(self.info["generator_config"]["output_mode"]).lower()

        # Reuse file structure library from a base dataset if it points to the same root
        if source_dataset is not None:
            assert str(source_dataset.fsl.fs.root_dpath) == str(Path(root_dpath))
            self.fsl = source_dataset.fsl
        else:
            self.fsl = DatasetFileStructureSessionLibrary(
                root_dpath, version=Version(self.info["version"]), extension=img_ext
            )

        self.lock = Lock()
        self.data = {}
        # Ensure attribute exists before first use in __getitem__
        self.n_actions = None

        # Set up cache
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache_dpath = Path(cache_dpath)
            self.cache_dpath.mkdir(parents=True, exist_ok=True)
            self.cache_metadata_fpath = (
                self.cache_dpath / f"metadata_{self.name}_{self.fsl.version}.feather"
            )

        # Keep a reference to base metadata to enable split views and reuse.
        if source_dataset is not None and hasattr(source_dataset, "metadata_base"):
            self.metadata_base = source_dataset.metadata_base
        elif source_dataset is not None and hasattr(source_dataset, "metadata_full"):
            # Backward-compat for any code still using metadata_full
            self.metadata_base = source_dataset.metadata_base
        else:
            self.metadata_base = self.__create_metadata()

        self.metadata = self.__build_split(self.metadata_base, split, split_type)

        self.n_instances = len(list(self.metadata.keys()))
        self.max_size = max_size

        self.sessions_info = self._build_sessions_from_metadata()
        self._recompute_window_index()

    def _build_sessions_from_metadata(self):
        sessions_info = []
        for inst_id, df in self.metadata.items():
            for sess_id, group in df.groupby("session_id"):
                first_start = 0 if self.fsl.fs.start_frame_fpath is None else 1
                sessions_info.append(
                    {
                        "instance_id": inst_id,
                        "session_id": int(sess_id),
                        "group": group.reset_index(drop=True),
                        "first_start": first_start,
                    }
                )
        return sessions_info

    def _recompute_window_index(self):
        seq_len = self.seq_length_current
        n_per = []
        for sess in self.sessions_info:
            group = sess["group"]
            first = sess["first_start"]
            L = (
                min(len(group), self.clip_length)
                if self.clip_length > 0
                else len(group)
            )
            last_exclusive = L - seq_len
            if last_exclusive <= first - 1:
                n = 0
            else:
                n = math.ceil((last_exclusive - first) / self.seq_step)
            n_per.append(max(0, n))
        self.n_windows_per_session = n_per
        self.cum_windows = (
            np.cumsum(self.n_windows_per_session) if n_per else np.array([], dtype=int)
        )
        self.total_windows = (
            int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0
        )

    @staticmethod
    def __read_info(info_fpath):
        if info_fpath.exists():
            with open(info_fpath, "r") as json_file:
                info = json.load(json_file)
        else:
            log.w(f"Info file not found at {info_fpath}. Using default info.")
            info = {"info": {}}

        default_info = {
            "action_space": [1],
            "observation_space": None,
            "version": f"{DEFAULT_VERSION}.0",
            "name": "default",
        }

        for key in default_info:
            if key not in info:
                info[key] = default_info[key]

        return info

    def __get_n_actions(self):
        # build a map of discrete actions
        if self.n_actions is None:
            actions = list(self.metadata.values())[0]["action"].to_numpy()
            # if isinstance(actions[0], list):
            self.n_actions = len(actions[0])
            # else:
            #     unique_top_left_values = np.unique(actions)
            #     unique_top_left_values.sort()
            #     # self.n_actions = len(unique_top_left_values)
            #     self.n_actions = 7

        return self.n_actions

    def _one_hot(self, x, n):
        try:
            return np.eye(n)[np.array(x).flatten().astype(int)]
        except IndexError:
            return np.eye(n)[np.zeros_like(x).flatten().astype(int)]

    def __build_split(self, metadata, split, split_type="instance"):
        n_elements = len(metadata)

        if split_type == "instance":
            dataset_split = range(
                int(math.floor(n_elements * self.__DATASET_SPLIT[split][0])),
                int(math.floor(n_elements * self.__DATASET_SPLIT[split][1])),
            )
            new_metadata = {id: metadata[id] for id in dataset_split}

        elif split_type == "session":
            new_metadata = {}
            for i in range(n_elements):
                session_ids = sorted(list(metadata[i]["session_id"].unique()))
                session_ids = session_ids[
                    int(
                        math.floor(len(session_ids) * self.__DATASET_SPLIT[split][0])
                    ) : int(
                        math.floor(len(session_ids) * self.__DATASET_SPLIT[split][1])
                    )
                ]
                new_metadata[i] = metadata[i][
                    metadata[i]["session_id"].isin(session_ids)
                ]

        elif split_type == "frame":
            new_metadata = {}
            for i in range(n_elements):
                new_metadata[i] = metadata[i].iloc[
                    int(
                        math.floor(
                            len(metadata[i].index) * self.__DATASET_SPLIT[split][0]
                        )
                    ) : int(
                        math.floor(
                            len(metadata[i].index) * self.__DATASET_SPLIT[split][1]
                        )
                    )
                ]

        else:
            raise ValueError(f"Invalid split type: {split_type}")

        return new_metadata


    def __create_metadata(self):
        has_read_error = False
        if self.enable_cache and self.cache_metadata_fpath.exists():
            self.log.d("Loading metadata from cache... : ", self.cache_metadata_fpath)
            with open(self.cache_metadata_fpath, "rb") as f:
                try:
                    metadata = {}
                    metadata_from_file = pd.read_feather(f)
                    for instance_id in metadata_from_file["instance_id"].unique():
                        metadata[instance_id] = metadata_from_file[
                            metadata_from_file["instance_id"] == instance_id
                        ]

                except (EOFError, OSError, Exception) as e:
                    self.log.e(
                        f"Error loading metadata from cache: {self.cache_metadata_fpath} ({type(e).__name__}: {e})"
                    )
                    has_read_error = True
        if (
            not (self.enable_cache and self.cache_metadata_fpath.exists())
            or has_read_error
        ):
            metadata = {}
            for session in tqdm(self.fsl, disable=len(self.fsl) <= 150):
                # Rest of the code inside the loop

                sessions = []
                for key in session.get_session_ids():
                    with open(
                        session.get_action_fpath(
                            session.instance_id, session_id=int(key)
                        ),
                        "r",
                    ) as json_file:
                        actions = json.load(json_file)
                        if "actions" in actions:
                            actions = actions["actions"]

                    session_pd = pd.DataFrame(actions)
                    session_pd["session_id"] = int(key)
                    sessions.append(session_pd)
                metadata_entry = pd.concat(sessions, axis=0, ignore_index=True)

                src_id_tag = (
                    "src_frame_id"
                    if "src_frame_id" in metadata_entry.columns
                    else "src_id"
                )
                tgt_id_tag = (
                    "tgt_frame_id"
                    if "tgt_frame_id" in metadata_entry.columns
                    else "tgt_id"
                )

                # metadata_entry["action"] = "click"

                # delete any alements where src_id is -1
                metadata_entry = metadata_entry[metadata_entry[src_id_tag] != -1]
                metadata_entry = metadata_entry[metadata_entry[tgt_id_tag] != -1]
                metadata_entry["src_frame_id"] = metadata_entry[src_id_tag].astype(
                    np.int32
                )
                metadata_entry["tgt_frame_id"] = metadata_entry[tgt_id_tag].astype(
                    np.int32
                )
                metadata_entry["session_id"] = metadata_entry["session_id"].astype(
                    np.int32
                )
                # metadata_entry["is_winning"] = metadata_entry["is_winning"].astype(bool)

                metadata_entry = metadata_entry.sort_values(
                    by=["session_id", "src_frame_id"]
                )

                metadata[session.instance_id] = metadata_entry

            if len(metadata) == 0:
                raise ValueError("Empty dataset!")

            if self.enable_cache:
                with open(self.cache_metadata_fpath, "wb") as f:
                    try:
                        metadata_to_store = pd.concat(
                            [
                                metadata[key].assign(instance_id=key)
                                for key in metadata.keys()
                            ],
                            ignore_index=True,
                        )
                        pd.DataFrame(metadata_to_store).to_feather(f)
                    except EOFError:
                        self.log.e(
                            f"EOFError writing metadata to cache: {self.cache_metadata_fpath}"
                        )
                    except OSError:
                        self.log.e(
                            f"OSError writing metadata to cache: {self.cache_metadata_fpath}"
                        )
                    finally:
                        pass

        return metadata

    def _read_frames(
        self,
        fpath,
        start_frame_fpath,
        frame_ids,
        scaling_factor=1,
        transform=None,
        output_mode="frame",
    ):
        """Read frames from a folder."""

        # If dataset was generated as a video, read frames from the session video
        if output_mode == "video":
            # Derive session directory from the formatted frame path (which points to .../frames/{:06d}.jpg)

            video_fpath = Path(fpath)

            cap = cv2.VideoCapture(str(video_fpath))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_fpath}")

            frames_list = []
            for frame_id in frame_ids:
                # Seek to the exact frame index (keyframe every frame in our generator)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    raise ValueError(
                        f"Could not read frame {frame_id} from video {video_fpath}."
                    )

                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1] * scaling_factor),
                        int(frame.shape[0] * scaling_factor),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

                if self.occlusion_mask is not None:
                    frame = (
                        frame * (1 - self.occlusion_mask) + self.occlusion_mask * 128
                    )

                if transform is not None:
                    frame = transform(frame)
                frames_list.append(frame)

            cap.release()
            frames = np.stack(frames_list, axis=0)
            if self.format == DatasetOutputFormat.PVG:
                frames = frames.transpose(0, 3, 1, 2)
            return frames

        # Default path: read from per-frame images
        def worker(frame_id, frame_fpath, scaling_factor=1, transform=None):
            """Read a single frame from a file."""
            frame = cv2.imread(str(frame_fpath).format(frame_id))
            if frame is None:
                raise ValueError(
                    f"Could not read frame {frame_id} from {str(frame_fpath).format(frame_id)}."
                )

            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * scaling_factor),
                    int(frame.shape[0] * scaling_factor),
                ),
                interpolation=cv2.INTER_AREA,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

            if self.occlusion_mask is not None:
                frame = frame * (1 - self.occlusion_mask) + self.occlusion_mask * 128

            if transform is not None:
                transformed_frame = transform(frame)
            else:
                transformed_frame = frame
            return transformed_frame

        max_workers = cpu_count()
        with ThreadPool(max_workers) as pool:
            frames = pool.starmap(
                worker,
                [
                    (
                        frame_id,
                        (
                            fpath
                            if frame_id != 0 or start_frame_fpath is None
                            else start_frame_fpath
                        ),
                        scaling_factor,
                        transform,
                    )
                    for frame_id in frame_ids
                ],
            )

        frames = np.stack(frames, axis=0)
        if self.format == DatasetOutputFormat.PVG:
            frames = frames.transpose(0, 3, 1, 2)
        return frames

    def __getitem__(self, idx):
        if self.total_windows == 0:
            raise IndexError("Empty dataset after indexing")

        if self.total_windows <= idx:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.total_windows}"
            )

        # idx = idx % self.total_windows
        sess_idx = int(np.searchsorted(self.cum_windows, idx + 1, side="left"))
        prev = 0 if sess_idx == 0 else int(self.cum_windows[sess_idx - 1])
        k = idx - prev
        sess = self.sessions_info[sess_idx]
        instance_id = sess["instance_id"]
        session_id = sess["session_id"]
        seq_start = sess["first_start"] + k * self.seq_step
        group = sess["group"]
        seq_metadata_entry = group.iloc[
            seq_start : seq_start + self.seq_length_current, :
        ]
        src_frame_ids = seq_metadata_entry["src_frame_id"].tolist()[
            : self.seq_length_current
        ]
        tgt_frame_id = seq_metadata_entry["tgt_frame_id"].tolist()[
            self.seq_length_current - 1
        ]
        actions = seq_metadata_entry["action"].to_list()[: self.seq_length_current]

        start_frame_fpath = self.fsl[instance_id].start_frame_fpath
        if self.output_mode == "frame":
            frame_fpath = self.fsl[instance_id].get_frame_fpath(
                session_id=session_id, session_props=None, frame_id=None
            )

            frames = self._read_frames(
                frame_fpath,
                start_frame_fpath,
                src_frame_ids + [tgt_frame_id],
                scaling_factor=1,
                transform=(
                    self.transform if self.format != DatasetOutputFormat.PVG else None
                ),
                output_mode=self.output_mode,
            )
        elif self.output_mode == "video":
            video_fpath = self.fsl[instance_id].get_video_fpath(
                session_id=session_id, session_props=None
            )
            frames = self._read_frames(
                video_fpath,
                start_frame_fpath,
                src_frame_ids + [tgt_frame_id],
                scaling_factor=1,
                transform=(
                    self.transform if self.format != DatasetOutputFormat.PVG else None
                ),
                output_mode=self.output_mode,
            )

        # frames = self.jitter_transform(torch.tensor(frames, dtype=torch.float32)).numpy()

        input_frames = frames[:-1]
        output_frame = frames[-1]

        actions = np.array([action for action in actions], dtype=np.uint8)
        if self.format == DatasetOutputFormat.IVG:
            is_first = np.zeros((self.seq_length_current + 1, 1), dtype=int)
            is_first[0, :] = np.ones_like(is_first[0, :])
            if np.array(self.info["info"]["action_space"])[-1] > 1:
                if (
                    self.n_actions is not None
                    and np.array(self.info["info"]["action_space"])[-1]
                    != self.n_actions
                ):
                    actions_one_hot = actions
                else:
                    actions_one_hot = actions
            else:
                actions_one_hot = self._one_hot(actions, self.__get_n_actions())

            if np.array(self.info["info"]["action_space"])[-1] == 7:
                actions_one_hot = self._one_hot(
                    actions, np.array(self.info["info"]["action_space"])[-1]
                )

                actions_one_hot = actions_one_hot[:, [1, 2, 0, 6, 3, 4, 5]]
                new_actions = []
                for action in actions_one_hot:
                    if action[-2] == 1:
                        action[0] = 1
                        action[4] = 1
                    elif action[-1] == 1:
                        action[1] = 1
                        action[4] = 1
                    new_actions.append(action)
                actions_one_hot = np.array(new_actions)
                actions_one_hot = actions_one_hot[:, :-2]

            output = {
                "input_frames": np.concatenate(
                    [input_frames, np.array([output_frame])], axis=0
                ).astype(np.float32),
                "output_frame": np.array([output_frame]),
                "actions": np.array(
                    actions_one_hot, dtype=np.float32
                ),  # np.array(actions, dtype=int),
                "is_first": is_first,
                "instance_id": instance_id,  # np.array(self._one_hot(instance_id, self.n_instances), dtype=np.float32),
                "n_instances": self.n_instances,
                "src_frame_ids": np.array(src_frame_ids),
                "tgt_frame_id": np.array([tgt_frame_id]),
            }

        elif self.format == DatasetOutputFormat.VICREG:
            actions = actions.reshape(-1, 1)
            rand_id = random.randint(0, len(input_frames))
            random_image = (input_frames[1:] + [output_frame])[rand_id]

            output = {
                "input_frames": np.concatenate(
                    [input_frames[[0]], np.array([random_image])]
                ).astype(np.float32),
                "instance_id": np.array(
                    self._one_hot(instance_id, self.n_instances), dtype=np.float32
                ),
            }
        elif self.format == DatasetOutputFormat.LAM:
            output = np.concatenate(
                [input_frames, np.array([output_frame])], axis=0
            ).astype(np.float32)

        else:
            output = {
                "input_frames": np.array(input_frames),
                "output_frame": np.array([output_frame]),
                "actions": np.array(actions),
                "src_frame_ids": np.array(src_frame_ids),
                "tgt_frame_id": np.array([tgt_frame_id]),
            }

            if self.enable_test:
                output["frame_ids"] = src_frame_ids + [tgt_frame_id]

        return output

    def __len__(self):
        size = self.total_windows if hasattr(self, "total_windows") else 0
        if self.max_size is not None and size > self.max_size:
            size = self.max_size
        return size

    def sample_actions(self, forbidden_actions):
        actions = torch.zeros_like(forbidden_actions)
        n_seq = forbidden_actions.shape[1]
        n_actions = forbidden_actions.shape[2]
        for i in range(n_seq):
            forbidden_action = forbidden_actions[0, i]
            while True:
                action = self._one_hot(random.randint(0, n_seq - 1), n_actions)
                if forbidden_action != action:
                    break
            actions[0, i, :] = torch.from_numpy(action)

        return actions


class TransformsGenerator:
    @staticmethod
    def color_jitter_transform(
        image: np.ndarray, brightness: int, contrast: int, saturation: int, hue: int
    ):
        """
        Apply color jitter to the image
        :param image: The image to apply color jitter to
        :param brightness: The brightness factor
        :param contrast: The contrast factor
        :param saturation: The saturation factor
        :param hue: The hue factor
        :return: The image with color jitter applied
        """
        image = image.transpose(0, 2, 3, 1)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Apply brightness jitter
        if brightness > 0:
            factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            image = np.clip(image * factor, 0, 1)

        # Apply contrast jitter
        if contrast > 0:
            factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * factor + mean, 0, 1)

        # Apply saturation jitter
        if saturation > 0:
            factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            gray = np.dot(image, [0.2989, 0.5870, 0.1140])
            gray = np.expand_dims(gray, axis=-1)
            image = np.clip(image * factor + gray * (1 - factor), 0, 1)

        # Apply hue jitter
        if hue > 0:
            shift_degrees = random.uniform(-hue, hue)
            for i in range(image.shape[0]):  # Iterate over the batch dimension
                # Convert to float first, but do not divide the hue by 255, just S and V:
                hsv_image = cv2.cvtColor(image[i], cv2.COLOR_RGB2HSV).astype(np.float32)

                # Extract the channels
                H = hsv_image[..., 0]
                S = hsv_image[..., 1] / 255.0
                V = hsv_image[..., 2] / 255.0

                # Convert the 8-bit Hue [0..179] to a [0..1] scale that represents 0..360 degrees
                # i.e. 179 -> 1.0 corresponds to 360 degrees
                # So H_float in [0..1] now means [0..360 deg].
                H_float = H / 179.0

                # Now apply your shift in "turns" or fraction of 1.0 for 360 degrees
                # e.g. want ±30 degrees => shift_fraction = ±(30/360)=±0.0833

                shift_fraction = shift_degrees / 360.0
                H_float = (H_float + shift_fraction) % 1.0

                # Convert back to OpenCV scale: if H_float=1.0 => 360 degrees => 179 in 8-bit
                H = (H_float * 179).astype(np.float32)
                S = (S * 255.0).astype(np.float32)
                V = (V * 255.0).astype(np.float32)

                hsv_image = np.stack([H, S, V], axis=-1).astype(np.uint8)
                image[i] = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        image = image.transpose(0, 3, 1, 2)
        return (image * 255).astype(np.uint8)

    @staticmethod
    def pad_to_match_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]):
        height, width = image.shape[:2]
        target_width, target_height = target_size

        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height

        # Determine padding
        if aspect_ratio > target_aspect_ratio:
            # Width is larger than target, pad top and bottom
            new_width = width
            new_height = int(width / target_aspect_ratio)
            top_pad = (new_height - height) // 2
            bottom_pad = new_height - height - top_pad
            pad_width = ((top_pad, bottom_pad), (0, 0), (0, 0))
        else:
            # Height is larger than target, pad left and right
            new_height = height
            new_width = int(height * target_aspect_ratio)
            left_pad = (new_width - width) // 2
            right_pad = new_width - width - left_pad
            pad_width = ((0, 0), (left_pad, right_pad), (0, 0))

        # Pad the image
        if len(image.shape) == 3:
            # Image has channels (e.g., RGB)
            padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)
        else:
            # Grayscale image
            pad_width = pad_width[:2]
            padded_image = np.pad(image, pad_width, mode="constant", constant_values=0)

        return padded_image

    @staticmethod
    def check_and_resize(
        target_crop: None | List[int], target_size: None | Tuple[int, int]
    ):
        """
        Creates a function that transforms input OpenCV images to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) tuple representing the target height and width
        :return: function that transforms an OpenCV image to the target size
        """

        # Creates the transformation function
        def transform(image: np.ndarray):
            if target_crop is not None:
                left, upper, right, lower = target_crop
                image = image[upper:lower, left:right]
            if target_size is not None and not all(
                dim == size for dim, size in zip(image.shape[:2], target_size)
            ):
                image = TransformsGenerator.pad_to_match_aspect_ratio(
                    image, target_size
                )
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def get_evaluation_transforms_config(config) -> Tuple:
        return TransformsGenerator.get_evaluation_transforms(
            config.data.crop, config.observation_space[config.enc_cnn_keys[0]]
        )

    @staticmethod
    def get_evaluation_transforms(crop_size, observation_space) -> Tuple:
        """
        Obtains the transformations to use for the evaluation scripts
        :param config: The evaluation configuration file
        :return: reference_transformation, generated transformation to use for the reference and the generated datasets
        """

        reference_resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )
        generated_resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )

        # Do not normalize data for evaluation
        reference_transform = transforms.Compose(
            [
                reference_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ]
        )
        generated_transform = transforms.Compose(
            [
                generated_resize_transform,
                transforms.ToTensor(),
                TransformsGenerator.to_float_tensor,
            ]
        )

        return reference_transform, generated_transform

    @staticmethod
    def get_final_transforms_config(config) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """

        return TransformsGenerator.get_final_transforms(
            config.observation_space[config.encoder.enc_cnn_keys[0]][1:],
            config.data.crop,
        )

    @staticmethod
    def get_final_transforms(
        observation_space, crop_size, **kwargs
    ) -> Dict[str, transforms.Compose]:
        """
        Obtains the transformations to use for training and evaluation

        :param config: The configuration file
        :type config: Config
        :param device: The device to use for computation
        :type device: torch.device
        :return: A dictionary containing the transformations for different stages
        :rtype: Dict[str, transforms.Compose]
        """
        # resize_transform = TransformsGenerator.check_and_resize(config.data.crop,
        #                                                         config.observation_space[config.encoder.enc_cnn_keys[0]][1:])

        components = []
        resize_transform = TransformsGenerator.check_and_resize(
            crop_size, observation_space
        )
        components.append(resize_transform)

        if "color_jitter" in kwargs:
            color_jitter_transform = transforms.Lambda(
                lambda x: TransformsGenerator.color_jitter_transform(
                    x,
                    kwargs["color_jitter"]["brightness"],
                    kwargs["color_jitter"]["contrast"],
                    kwargs["color_jitter"]["saturation"],
                    kwargs["color_jitter"]["hue"],
                )
            )
            components.append(color_jitter_transform)

        components.append(transforms.ToTensor())

        transform = transforms.Compose(components)

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }


class EnvironmentDatasetTester:
    def __init__(self, dataset: EnvironmentDataset, ouput_dpath):
        self.dataset = dataset
        self.output_dpath = ouput_dpath
        self.output_dpath.mkdir(parents=True, exist_ok=True)
        for f in self.output_dpath.iterdir():
            f.unlink()

    def visualize(self, idx):
        data = self.dataset[idx]
        input_frames = list(data["input_frames"])
        for i in range(len(input_frames)):
            input_frames[i] = cv2.resize(
                input_frames[i],
                (input_frames[i].shape[1] * 4, input_frames[i].shape[0] * 4),
            )

        input_frames = np.array(input_frames)

        output_frame = data["output_frame"][0]
        output_frame = cv2.resize(
            output_frame, (output_frame.shape[1] * 4, output_frame.shape[0] * 4)
        )
        actions = data["actions"]
        # output_frame = output_frame.transpose(0, 1, 2, 0)
        # input_frames = input_frames.transpose(0, 2, 3, 1)

        actions_captions = []
        if "action_captions" in self.dataset.info["info"]:
            captions = self.dataset.info["info"]["action_captions"]

            for i in range(len(actions)):
                actions_captions.append(captions[np.argmax(actions[i])])
        else:
            actions_captions = actions.tolist()

        # #in output, on each image across the first dimension add text with the frame_id

        for i in range(self.dataset.seq_length_current):
            cv2.putText(
                input_frames[i, :, :, :],
                str(data["src_frame_ids"][i]) + f": {actions_captions[i]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            output_frame,
            str(data["tgt_frame_id"]),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # concatenate all images in input_frames
        input_frames = np.concatenate(input_frames, axis=1)
        output = np.concatenate([input_frames, output_frame], axis=1)

        log.i("Writing to", str(self.output_dpath / f"{idx}_output.png"))
        cv2.imwrite(str(self.output_dpath / f"{idx}_output.png"), output[:, :, ::-1])

    def __getitem__(self, idx):
        self.visualize(idx)
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tester = EnvironmentDatasetTester(
        EnvironmentDataset(
            "../../datasets/retro_act_v1.1.1/retro_montezumarevenge-atari2600_v1.1.1__frameskip4/",
            seq_length_input=15,
            split="test",
            split_type="instance",
        ),
        Path("output/retro_act_v1.0.1"),
    )
    for idx, i in enumerate(tester):
        pass
