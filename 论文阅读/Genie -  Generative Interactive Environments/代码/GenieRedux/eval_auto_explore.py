import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.model_management import CheckpointDirManager
from auto_explore.src.utils import dump_hydra, FileStructure
from auto_explore.src.trainer import Trainer
import argparse

if "SLURM_NTASKS" in os.environ:
    # Remove SLURM env variables to avoid issues with Lightning
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]
    
from lightning.fabric import Fabric
from tools.logger import getLogger
log = getLogger(__name__)

def run(cfg: DictConfig):
    cfg.common.root_dpath = os.path.abspath(cfg.common.root_dpath)
    cfg.world_model.root_dpath = os.path.abspath(cfg.world_model.root_dpath)

    model_id = cfg.common.resume_id
    cdm = CheckpointDirManager(cfg.common.root_dpath)
    dpath = cdm.get_dpath_by_id(model_id)
    
    fabric = Fabric(strategy="ddp", accelerator=cfg.common.device, devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.barrier()

    log.i("Evaluatin model id: ", model_id, dpath)

    fname = dpath.name
    cfg.wandb.name = fname + "_eval"
    cfg.common.resume=True
    dpath = Path(dpath)
    os.makedirs(dpath, exist_ok=True)
    os.chdir(dpath)

    log.i(f"Running experiment: {fname}")
    fs = FileStructure(dpath)
    fs.create()
    dump_hydra(cfg, fs.hydra_config_fpath)
    trainer = Trainer(cfg, fabric, fs)
    # In evaluation, interpret `common.epochs` as "how many eval epochs to run",
    # not as an absolute target epoch (avoid skipping to last trained epoch).
    try:
        trainer.start_epoch = 1
    except Exception:
        pass
    # Enable per-epoch and final evaluation summaries only for eval runs
    try:
        cfg.evaluation.print_summary = True
    except Exception:
        pass
    trainer.run()


@hydra.main(config_path="auto_explore/configs", config_name="evaluate")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
