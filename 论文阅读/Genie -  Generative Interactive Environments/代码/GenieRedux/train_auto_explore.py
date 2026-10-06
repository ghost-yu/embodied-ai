import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.model_management import CheckpointDirManager
from auto_explore.src.utils import dump_hydra, FileStructure
from auto_explore.src.trainer import Trainer

if "SLURM_NTASKS" in os.environ:
    # Remove SLURM env variables to avoid issues with Lightning
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]
    
from lightning.fabric import Fabric
from tools.logger import getLogger
log = getLogger(__name__)


def run(cfg: DictConfig):
    fabric = Fabric(strategy="ddp", accelerator=cfg.common.device, devices="auto", precision="bf16-mixed")
    fabric.launch()
    fabric.barrier()
    cfg.common.root_dpath = os.path.abspath(cfg.common.root_dpath)
    cfg.world_model.root_dpath = os.path.abspath(cfg.world_model.root_dpath)

    root_dpath = cfg.common.root_dpath

    cdm = CheckpointDirManager(root_dpath)
    if cfg.common.resume:
        dpath = cdm.get_last_dpath()
    else:
        if fabric.is_global_zero:
            dpath = cdm.build_dpath_next(cfg.common.name)
        
        fabric.barrier()

        if not fabric.is_global_zero:
            cdm.update()
            dpath = cdm.get_last_dpath()

    
    log.i(f"Using wm {cfg.world_model.model_dname}")

    fname = dpath.name
    cfg.wandb.name = fname

    dpath = Path(dpath)
    os.makedirs(dpath, exist_ok=True)
    os.chdir(dpath)

    log.i(f"Running experiment: {fname}")
    fs = FileStructure(dpath)
    log.i("Data path:", dpath)
    fs.create()
    dump_hydra(cfg, fs.hydra_config_fpath)
    trainer = Trainer(cfg, fabric, fs)
    trainer.run()


@hydra.main(config_path="auto_explore/configs", config_name="trainer")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
