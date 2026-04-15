import collections
import logging
import os
import sys
from pathlib import Path

import yaml


def parse_config(config):
    try:
        collections_abc = collections.abc
    except AttributeError:
        collections_abc = collections

    if isinstance(config, collections_abc.Mapping):
        return config

    assert isinstance(config, str)

    if not os.path.exists(config):
        raise IOError(
            "config path {} does not exist. "
            "Please pass in a dict or a valid yaml file path.".format(config)
        )

    with open(config, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def expand_path(path_value):
    return Path(os.path.expandvars(str(path_value)))


def setup_ei_tuning_logger(task, wtype, parc, pat):
    log_dir = Path(f"logs/{task}/{wtype}/parc{parc}")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{pat}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger(__name__)
