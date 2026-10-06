import copy
import json

import tqdm
from generator.generator import EnvironmentDataGenerator

from omegaconf import OmegaConf
import hydra


def run_env(config, connector_config):
    connector_class_name = connector_config["classname"]
    del connector_config["classname"]
    generator_config = connector_config["generator_config"]
    del connector_config["generator_config"]

    generator = EnvironmentDataGenerator(
        connector_class_name,
        connector_config,
        generator_config,
        config,
    )
    print(f"Generating {generator.name}...")
    generator.generate()
    print(f"Done with {generator.name}...")


def hydra_cfg_to_legacy_dict(cfg) -> dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Hydra config did not resolve to a dict.")
    # normalize hydra-composed config to legacy dict
    # Option B: namespaced 'connector' node (preferred)
    if "connector" in cfg_dict and isinstance(cfg_dict["connector"], dict):
        connector = cfg_dict["connector"]
        env = connector.get("env") or cfg_dict.get("env")
        if env is None:
            raise ValueError("Hydra config 'connector' missing required key 'env' and no top-level 'env' found.")
        return {
            "data_dpath": cfg_dict.get("data_dpath"),
            "dname": cfg_dict.get("dname", ""),
            "env": env,
            f"connector_{env}": connector,
        }

    # Some compositions may place the preset under a top-level 'config' node
    # (e.g., if the group was injected with a package or via override). Handle that too.
    if "config" in cfg_dict and isinstance(cfg_dict["config"], dict):
        inner = cfg_dict["config"]
        if "connector" in inner and isinstance(inner["connector"], dict):
            connector = inner["connector"]
            env = connector.get("env") or cfg_dict.get("env")
            if env is None:
                raise ValueError("Hydra config under 'config' missing 'connector.env' and no top-level 'env'.")
            # Prefer dname from nested config block; fall back to root if provided
            dname = inner.get("dname", cfg_dict.get("dname", ""))
            return {
                "data_dpath": cfg_dict.get("data_dpath"),  # stays at root
                "dname": dname,
                "env": env,
                f"connector_{env}": connector,
            }

    # Fallback: flat root composition
    env = cfg_dict.get("env")
    if env is None:
        raise ValueError("Hydra config missing required key 'env'.")
    global_keys = {"data_dpath", "dname", "hydra"}
    connector = {k: v for k, v in cfg_dict.items() if k not in global_keys}
    return {
        "data_dpath": cfg_dict.get("data_dpath"),
        "dname": cfg_dict.get("dname", ""),
        "env": env,
        f"connector_{env}": connector,
    }

@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(cfg):
    config = hydra_cfg_to_legacy_dict(cfg)

    connector_configs = []
    if config["env"] == "coinrun":
        connector_config = config["connector_" + config["env"]]

        from generator.connector_coinrun import CoinRunConnector

        connector_config["classname"] = CoinRunConnector

        connector_configs.append(connector_config)

    elif config["env"] == "retro_act":
        connector_config_retro_act = config["connector_" + config["env"]]

        from generator.connector_retro_act import (
            GameData,
            RetroActConnector,
            RetroActAutoExploreConnector,
            RetroActAgent57Connector,
        )

        # Choose connector class by variant (normalize missing/legacy 'default' -> 'random')
        variant = connector_config_retro_act.get("variant", "random")
        if variant == "default":
            variant = "random"
        if variant == "auto_explore":
            connector_cls = RetroActAutoExploreConnector
        elif variant == "agent57":
            connector_cls = RetroActAgent57Connector
        elif variant == "random":
            connector_cls = RetroActConnector

        connector_config_retro_act["classname"] = connector_cls
        game_data = GameData(
            annotation_fpath=connector_config_retro_act["annotation_behavior_fpath"],
            control_annotation_fpath=connector_config_retro_act[
                "annotation_control_fpath"
            ],
        )

        selected_games = connector_config_retro_act["game"]
        motion_filter = connector_config_retro_act["motion"]
        view_filter = connector_config_retro_act["view"]
        genre_filter = connector_config_retro_act["genre"]
        platform_filter = connector_config_retro_act["platform"]
        action_map = connector_config_retro_act["action_map"]

        if action_map == "default":
            action_map = {
                "ACTION_JUMP": "jump",
                "DOWN": "none|crouch|climb",
                "UP": "climb|none",
                "LEFT": "left",
                "RIGHT": "right",
            }
            
        game_data.filter(action_map=action_map)
        game_filter = None if selected_games == "all" else selected_games

        selected_games = game_data.query(
            genre=genre_filter,
            motion=motion_filter,
            view=view_filter,
            game=game_filter,
            platform=platform_filter,
        )

        if len(selected_games) == 0:
            raise ValueError(
                f"No games found for genre={genre_filter}, motion={motion_filter}, view={view_filter}, platform={platform_filter}",
            )

        # Optionally sort alphabetically and limit to first N titles when requested
        limit_games = connector_config_retro_act.get("limit_games", None)
        selected_games = sorted(selected_games)
        if isinstance(limit_games, int) and limit_games > 0:
            selected_games = selected_games[:limit_games]

        print(
            f"Found {len(selected_games)} games for genre={genre_filter}, motion={motion_filter}, view={view_filter}",
        )

        for i, game in enumerate(selected_games):
            connector_config_retro_act_temp = copy.deepcopy(connector_config_retro_act)
            connector_config_retro_act_temp["game"] = game
            # if "PowerPiggs" in game:
            connector_configs.append(connector_config_retro_act_temp)
    else:
        raise ValueError(f"Unknown environment {config['env']}")

    for connector_config in tqdm.tqdm(connector_configs):
        run_env(config, connector_config)

    # pool = multiprocessing.Pool(processes=min(len(connector_configs),args.max_workers))
    # pool.starmap(run_env, [(config, connector_config) for connector_config in connector_configs])
    # pool.close()
    # pool.join()
    print("Done")


if __name__ == "__main__":
    main()
