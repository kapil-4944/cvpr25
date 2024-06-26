# Shree KRISHNAya Namaha
# Common utility functions
# Author: Nagabhushan S N
# Last Modified: 29/12/23
import importlib
import time
import datetime
import traceback
from typing import Dict, Any

import numpy
import simplejson
import skimage.io
import skvideo.io
import pandas

from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot

this_filepath = Path(__file__)
this_filename = this_filepath.stem


def get_test_videos_datapath(database_dirpath: Path, pred_train_dirpath: Path):
    train_configs_path = pred_train_dirpath / 'Configs.py'
    train_configs = read_configs(train_configs_path)
    set_num = train_configs['set_num']
    test_videos_datapath = database_dirpath / f'train_test_sets/set{set_num:02}/TestVideosData.csv'
    return test_videos_datapath


def read_configs(configs_path: Path):
    spec = importlib.util.spec_from_file_location(configs_path.stem, configs_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    configs: Dict[str, Any] = cfg.config
    return configs
