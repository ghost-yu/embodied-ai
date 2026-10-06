import os
from pathlib import Path
import shutil
from typing import List
from natsort import natsorted


class CheckpointManager:
    ID="ID"
    METRIC="METRIC"
    ITER="ITER"
    
    def __init__(self, checkpoints_root_dpath: Path | str, patfname: str = "$ID_model_it$ITER_eval$METRIC.pt") -> None:
        if type(checkpoints_root_dpath) is str:
            self.checkpoints_root_dpath = Path(checkpoints_root_dpath)
        else:
            self.checkpoints_root_dpath = checkpoints_root_dpath
        self.checkpoints_root_dpath.mkdir(parents=True, exist_ok=True)
        self.update()

    def update(self):
        self.checkpoints_fnames = natsorted(os.listdir(self.checkpoints_root_dpath))
        self.checkpoints_fnames = list(
            filter(lambda fname: "checkpoint" in fname or "model" in fname or fname.endswith('.pt'), self.checkpoints_fnames)
        )
        self.checkpoints_dict = {}
        for fname in self.checkpoints_fnames:
            if "_" in fname:
                id = int(fname.rsplit(".", 1)[0].split("_", 1)[1])
                if id not in self.checkpoints_dict:
                    self.checkpoints_dict[id] = []
                self.checkpoints_dict[id].append(fname)

    def log_state(self):
        ids = self.checkpoints_dict.keys()
        if len(ids) == 0:
            print("No checkpoints found!")
            return

        # print min and max ids
        print(f"Min checkpoint id: {min(ids)}")
        print(f"Max checkpoint id: {max(ids)}")

        repeating_ids = [
            id for id, lst in self.checkpoints_dict.items() if len(lst) > 1
        ]
        if len(repeating_ids) > 0:
            print(f"Repeating checkpoints: {repeating_ids}")

    def get_fpath_by_id(self, id: int) -> Path:
        if id not in self.checkpoints_dict:
            raise Exception(f"Checkpoint file with id {id} not found!")

        if len(self.checkpoints_dict[id]) > 1:
            raise Exception(
                f"Multiple checkpoint files found with id {id}: {self.checkpoints_dict[id]}"
            )
        return self.checkpoints_root_dpath / self.checkpoints_dict[id][0]

    def get_last_fpath(self) -> Path:

        if len(self.checkpoints_dict) == 0:
            raise Exception(
                f"No checkpoint files found at {self.checkpoints_root_dpath}"
            )
        return self.get_fpath_by_id(max(self.checkpoints_dict.keys()))

    def get_last_id(self) -> int:
        return max(self.checkpoints_dict.keys())

    def check_id_existence(self, id: int) -> bool:
        return id in self.checkpoints_dict

    def delete_by_id(self, id: int, not_exist_ok: bool = True):
        if not self.check_id_existence(id):
            if not not_exist_ok:
                raise Exception(f"Checkpoint file with id {id} does not exist!")
            else:
                return

        fpath = self.get_fpath_by_id(id)
        os.remove(fpath)
        self.update()

    def keep_last_k(self, k: int):
        self.update()
        ids = sorted(list(self.checkpoints_dict.keys()))
        for id in ids[:-k]:
            self.delete_by_id(id)

    def keep_best_k(self, k: int, metric: float):
        self.update()
        ids = sorted(list(self.checkpoints_dict.keys()))
        best_ids = sorted(ids, key=lambda id: self.get_metric(id, metric_name))[-k:]
        for id in ids:
            if id not in best_ids:
                self.delete_by_id(id)


class CheckpointDirManager:
    def __init__(self, checkpoints_root_dpath: Path | str) -> None:
        if type(checkpoints_root_dpath) is str:
            self.checkpoints_root_dpath = Path(checkpoints_root_dpath)
        else:
            self.checkpoints_root_dpath = checkpoints_root_dpath

        self.temp_id = 0
        self.checkpoints_root_dpath.mkdir(parents=True, exist_ok=True)
        self.update()

    def update(self):
        self.checkpoints_dnames = natsorted(os.listdir(self.checkpoints_root_dpath))
        self.checkpoints_dnames = list(
            filter(
                lambda fname: fname.split("_", 1)[0].isdigit(), self.checkpoints_dnames
            )
        )
        
        self.checkpoints_dict = {}
        for fname in self.checkpoints_dnames:
            id = int(fname.split("_", 1)[0])
            if id not in self.checkpoints_dict:
                self.checkpoints_dict[id] = []
            self.checkpoints_dict[id].append(fname)

    def log_state(self):
        ids = self.checkpoints_dict.keys()
        if len(ids) == 0:
            print("No checkpoints found!")
            return

        # print min and max ids
        print(f"Min checkpoint id: {min(ids)}")
        print(f"Max checkpoint id: {max(ids)}")

        missing_ids = [i for i in range(min(ids), max(ids) + 1) if i not in ids]
        if len(missing_ids) > 0:
            print(f"Missing checkpoints: {missing_ids}")

        repeating_ids = [
            id for id, lst in self.checkpoints_dict.items() if len(lst) > 1
        ]
        if len(repeating_ids) > 0:
            print(f"Repeating checkpoints: {self.repeating_numbers}")

    def get_dpath_by_id(self, id: int) -> Path:
        if id not in self.checkpoints_dict:
            raise Exception(f"Checkpoint with id {id} not found!")

        if len(self.checkpoints_dict[id]) > 1:
            raise Exception(
                f"Multiple checkpoints found with id {id}: {self.checkpoints_dict[id]}"
            )
        return (self.checkpoints_root_dpath / self.checkpoints_dict[id][0]).resolve()

    def get_dpath_by_description(self, description: str) -> List[Path]:
        dpaths = []
        for dname in self.checkpoints_dnames:
            if description in dname:
                dpaths.append(self.checkpoints_root_dpath / dname)

        if not dpaths:
            raise Exception(f"No checkpoints found with description {description}!")

        if len(dpaths) > 1:
            raise Exception(
                f"Multiple checkpoints found with description {description}: {dpaths}"
            )

        return dpaths[0].resolve()

    def get_dpath_by_id_or_description(self, id_or_description: str) -> List[Path]:
        if id_or_description.isdigit():
            return self.get_dpath_by_id(int(id_or_description))
        else:
            return self.get_dpath_by_description(id_or_description)

    def get_last_dpath(self) -> Path:
        return self.get_dpath_by_id(max(self.checkpoints_dict.keys()))

    def get_last_id(self) -> int:
        if len(self.checkpoints_dict) == 0:
            raise Exception(f"No checkpoints found!")
        return max(self.checkpoints_dict.keys())

    def get_dpath_temp(self) -> Path:
        """
        Returns the temporary directory path of the last model.
        Returns:
            Path: The path to the temporary debugging model directory.
        """
        return self.get_dpath_by_id(self.temp_id)

    def build_dpath_temp(self) -> Path:
        """
        Generates and returns the temporary directory path for debugging purposes.
        This method constructs a model directory path specifically meant for debugging.
        The directory is identified by an ID of 0 and is labeled with the description "debug".
        Note that the weights inside this directory will be overwritten.
        Returns:
            Path: The path to the temporary debugging model directory.
        """
        try:
            temp_dpath = self.get_dpath_by_id(self.temp_id)
        except Exception:
            temp_dpath = self.build_dpath_by_id(self.temp_id, description="debug")
        return temp_dpath
    
    def build_dpath_by_id(
        self,
        id: int,
        description: str = "",
        base_id: int | None = None,
        base_iter: int | None = None,
    ) -> Path:
        # self.delete_empty_by_id(id)
        if id in self.checkpoints_dict:
            raise Exception(f"Checkpoint with id {id} already exists!")

        dname = f"{id:03}"
        if description:
            dname = dname + "_" + description

        if base_id is not None:
            dname = dname + "_base_" + f"{base_id:03}"

        if base_iter is not None:
            dname = dname + "_iter_" + f"{base_iter}"

        out_dpath = self.checkpoints_root_dpath / dname

        if base_id is not None:
            out_dpath.mkdir(parents=True, exist_ok=False)
            base_dpath = self.get_dpath_by_id(base_id)
            if base_iter is not None:
                base_fpath = CheckpointManager(base_dpath).get_fpath_by_id(base_iter)
            else:
                base_fpath = CheckpointManager(base_dpath).get_last_fpath()
            shutil.copy(base_fpath, out_dpath)

        out_dpath.mkdir(parents=True, exist_ok=True)
        self.update()
        return out_dpath.resolve()

    def get_next_id(self) -> int:
        return (
            self.get_last_id() + 1
            if len(self.checkpoints_dict) > 0
            else 1
        )

    def build_dpath_next(
        self, description="", base_id: int | None = None, base_iter: int | None = None
    ) -> Path:
        # self.delete_empty()
        next_id = self.get_next_id()
        return self.build_dpath_by_id(next_id, description, base_id, base_iter)

    def check_id_existence(self, id: int) -> bool:
        return id in self.checkpoints_dict

    def check_description_existence(self, description: str) -> bool:
        return any(description in dname for dname in self.checkpoints_dnames)

    def delete_by_id(self, id: int, not_exist_ok: bool = True):
        if not self.check_id_existence(id):
            if not not_exist_ok:
                raise Exception(f"Checkpoint with id {id} does not exist!")
            else:
                return

        dpath = self.get_dpath_by_id(id)
        shutil.rmtree(dpath)
        self.update()

    def delete_until_id(self, id: int):
        for id in range(0, id):
            try:
                self.delete_by_id(id)
            except:
                pass

        self.update()

    def is_empty(self, dpath: Path):
        return not [
            fname
            for fname in os.listdir(dpath)
            if not fname.endswith(".yaml") and fname not in ["log", "wandb"]
        ]

    def delete_empty_by_id(self, id: int, not_exist_ok: bool = True):
        if not self.check_id_existence(id):
            if not not_exist_ok:
                raise Exception(f"Checkpoint with id {id} does not exist!")
            else:
                return

        try:
            dpath = self.get_dpath_by_id(id)
        except:
            pass

        if self.is_empty(dpath):
            shutil.rmtree(dpath)
            self.update()

    def delete_empty(self, test_mode=False):
        """
        Deletes empty directories from the checkpoints directory. Keeps the last directory.

        Args:
            test_mode (bool, optional): If True, the deletion will be simulated without actually removing any directories.
                Defaults to False.
        """
        last_dpath = self.get_last_dpath()
        for dname in self.checkpoints_dnames:
            dpath = self.checkpoints_root_dpath / dname
            if dpath != last_dpath and self.is_empty(dpath):
                if not test_mode:
                    shutil.rmtree(dpath)
                else:
                    print(f"Empty directory: {dpath}")
                self.update()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpath", type=str, required=True)
    args = parser.parse_args()

    cdm = CheckpointDirManager(args.dpath)
    cdm.log_state()
    cdm.delete_empty(test_mode=False)
    # cdm.delete_empty()
    # cdm.delete_by_id(1)
    # cdm.delete_until_id(2)
    # cdm.delete_empty_by_id(2)
    # cdm.build_dpath_by_id(1)
