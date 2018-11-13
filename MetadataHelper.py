import yaml
import pandas as pd
import os

def loadSettings(settings_path):
    DEFAULT_PATH = "config/user_default.yml"
    with open(DEFAULT_PATH, 'r') as d:
        settings = yaml.load(d)

    with open(settings_path, 'r') as f:
        settings2 = yaml.load(f)
    
    settings.update(settings2)
    return(settings)

def loadExistingFlowGates(pickle_path):
    if os.path.isfile(pickle_path):
        all_stats = pd.read_pickle(pickle_path)
    else:
        all_stats = False
    return(all_stats)