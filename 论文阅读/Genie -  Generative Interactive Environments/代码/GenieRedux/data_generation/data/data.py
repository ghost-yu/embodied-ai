import json
import os
import os.path as osp
from pathlib import Path

from packaging.version import Version

DEFAULT_VERSION = Version("0.0.0")


class DatasetFileStructure:
    INSTANCE_ID = "{INSTANCE}"
    SESSION_ID = "{SESSION}"
    TIMESTAMP = "{TIMESTAMP}"
    WIN_LABEL = "win"
    LOSE_LABEL = "lose"

    def __init__(
        self,
        root_dpath: str,
        version: Version = DEFAULT_VERSION,
        extension="jpg",
    ) -> None:
        self.root_dpath = Path(root_dpath)
        # self.root_dpath.mkdir(parents=True, exist_ok=True)
        self.instance_id_format = "{:06d}"  # "{:04d}"
        self.session_id_format = "{:06d}"  # "{:04d}"
        self.frame_id_format = "{:06d}"
        self.version = version

        self.info_fpath = self.root_dpath / "info.json"

        if self.info_fpath.exists():
            with open(self.info_fpath) as json_file:
                info = json.load(json_file)
                self.version = Version(str(info["version"]))
                # if version < self.version:
                #     log.w("Warning. The dataset is outdated for this code.")

        self.instance_dpath: Path = self.root_dpath / self.INSTANCE_ID
        self.frame_fname = f"{self.frame_id_format}.{extension}"
        # the following is deprecated in version 2
        # if "minesweeper" in self.root_dpath.stem and  self.version < 2:
        #     self.session_win_dpath = self.instance_dpath / self.WIN_LABEL
        #     self.session_lose_dpath = self.instance_dpath / self.LOSE_LABEL
        #     self.debug_lose_dpath = self.session_lose_dpath / "debug"
        #     self.debug_win_dpath = self.session_win_dpath / "debug"
        #     self.frame_win_fpath = self.session_win_dpath / self.frame_fname
        #     self.frame_lose_fpath = self.session_lose_dpath / self.frame_fname
        # else:
        self.session_fmtdpath: Path = self.instance_dpath / self.SESSION_ID
        self.frame_fmtdpath: Path = self.session_fmtdpath / "frames"
        self.frame_fmtfpath: Path = self.frame_fmtdpath / self.frame_fname

        self.actions_fname = "actions.json"
        # self.actions_fpath = self.instance_dpath / self.actions_fname
        self.actions_fpath: Path = self.session_fmtdpath / self.actions_fname
        # self.start_frame_fpath = self.root_dpath / "start_frame.png"
        self.start_frame_fpath = None
        self.video_fmtdpath = self.session_fmtdpath / "frames.mp4"

    def get_action_fpath(self, instance_id, session_id, make_dirs=False):
        instance_id = int(instance_id)
        session_id = int(session_id)
        fpath = Path(
            str(self.actions_fpath)
            .replace(self.INSTANCE_ID, self.instance_id_format.format(instance_id))
            .replace(self.SESSION_ID, self.session_id_format.format(session_id)),
        )

        if make_dirs:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
        return fpath

    @staticmethod
    def get_ids_from_dpath(root_dpath: Path):
        """
        Returns the instance and session IDs from a file path.

        Parameters:
        - fpath (Path): The file path.

        Returns:
        - tuple: The instance and session IDs.
        """
        return [int(dpath.name) for dpath in root_dpath.iterdir() if dpath.is_dir()]

    def get_instance_ids(self) -> list[int]:
        """
        Returns a list of instance IDs contained in the root directory.

        Returns:
            list: A list of instance IDs.
        """
        return DatasetFileStructure.get_ids_from_dpath(self.root_dpath)


class DatasetFileStructureInstance(DatasetFileStructure):
    @staticmethod
    def replace(
        path: str | Path | None,
        old_str: str,
        new_str: str,
    ) -> str | Path | None:
        """Replace a string in a Path."""

        if path is None:
            return None

        if isinstance(path, str):
            new_path = path.replace(old_str, new_str)
        elif isinstance(path, Path):
            new_path = Path(str(path).replace(old_str, new_str))
        return new_path

    def __init__(
        self,
        root_dpath,
        instance_id,
        version: Version = DEFAULT_VERSION,
        **kwargs,
    ) -> None:
        super().__init__(root_dpath, version=version, **kwargs)
        self.instance_id = instance_id
        self.instance_id_label = self.instance_id_format.format(instance_id)
        self.version = version

        # Find all the variable names defined in the super class
        super_class_vars = [
            var_name for var_name in dir(super()) if not var_name.startswith("__")
        ]

        # Make a list of only the ones ending corresponding to file structure paths and names
        variable_names = [
            var_name
            for var_name in super_class_vars
            if isinstance(var_name, str)
            and var_name.endswith(
                ("_dpath", "_fpath", "_dname", "_fname", "_fmtfpath", "_fmtdpath"),
            )
        ]

        # Replace the instance ID in the variable names
        for var_name in variable_names:
            setattr(
                self,
                var_name,
                self.replace(
                    getattr(self, var_name),
                    self.INSTANCE_ID,
                    self.instance_id_label,
                ),
            )

        self.create_tree()

    def create_tree(self):
        """
        Creates directory tree based on the specified directory paths.

        This method retrieves all the directory paths from the instance variables of the class
        and creates the corresponding directories if they don't exist.

        Args:
            None

        Returns:
            None
        """
        class_vars = [
            var_name for var_name in dir(self) if not var_name.startswith("__")
        ]
        dpaths = [
            getattr(self, var_name)
            for var_name in class_vars
            if isinstance(var_name, str)
            and var_name.endswith("_dpath")
            and not callable(getattr(self, var_name))
        ]
        for dpath in dpaths:
            dpath.mkdir(parents=True, exist_ok=True)

    def get_frame_fpath(self, session_id, session_props, frame_id):
        """
        Returns the file path for a frame in an instance.

        Parameters:
        - session_id (str): The ID of the session.
        - session_props (dict): The properties of the session.
        - frame_id (int): The ID of the frame.

        Returns:
        - str: The file path of the frame.
        """

        # if self.version < 2:
        #     is_winning = session_props["is_winning"]
        #     if is_winning:
        #         dpath = self.session_win_dpath
        #     else:
        #         dpath = self.session_lose_dpath
        # else:
        session_id_label = self.session_id_format.format(session_id)
        dpath: Path = self.replace(
            self.frame_fmtdpath,
            self.SESSION_ID,
            session_id_label,
        )  # type: ignore
        dpath.mkdir(parents=True, exist_ok=True)

        if frame_id is not None:
            fname = self.frame_fname.format(frame_id)
        else:
            fname = self.frame_fname
        fpath = dpath / fname
        return str(fpath)

    def get_video_fpath(self, session_id, session_props):
        """
        Returns the file path for a video in an instance.

        Parameters:
        - session_id (str): The ID of the session.

        Returns:
        - str: The file path of the video.
        """

        session_id_label = self.session_id_format.format(session_id)
        fpath: Path = self.replace(
            self.video_fmtdpath,
            self.SESSION_ID,
            session_id_label,
        )  # type: ignore

        return str(fpath)

    def get_session_ids(self) -> list[int]:
        """
        Returns a list of session IDs contained in the instance directory.

        Returns:
            list: A list of session IDs.
        """
        return DatasetFileStructureInstance.get_ids_from_dpath(self.instance_dpath)
