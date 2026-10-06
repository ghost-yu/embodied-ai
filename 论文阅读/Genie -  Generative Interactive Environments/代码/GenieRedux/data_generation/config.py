# get the directory of the current file
import os
import pathlib
import sys

PROJECT_IVG_DATA_DPATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(PROJECT_IVG_DATA_DPATH))

DATASET_VERSION = 2.0


def update_dataset_version(new_version):
    global DATASET_VERSION
    global DATASET_MINESWEEPER_DPATH
    DATASET_VERSION = new_version
    DATASET_MINESWEEPER_DPATH = (
        PROJECT_IVG_DATA_DPATH / f"minesweeper_v{new_version:.1f}"
    )


update_dataset_version(DATASET_VERSION)
