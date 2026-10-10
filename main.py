import pandas as pd
import numpy as np

import os, json

CONFIG_LOCAL_FILE = "config_local.json"
CONFIG_FILE = "config.json"

if __name__ == "__main__":
    config = json.load(open(CONFIG_FILE, "r"))
    config_local = json.load(open(CONFIG_LOCAL_FILE, "r"))

    config = {config_local["root_dir"]+s if s.split('_')[-1] in ["file", "dir"] else s for s in config}

    