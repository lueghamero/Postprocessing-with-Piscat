from ParallelProcessingConfig import CPUConfig
from piscat.InputOutput import CPUConfigurations
import json
import os

import pandas as pd



def deletecpuconfig():
    subdir = "piscat_configuration"
    here = os.path.abspath(os.path.join(os.getcwd(), '..'))
    filepath = os.path.join(here, subdir, "cpu_configurations.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Configuration file {filepath} deleted successfully.")
        except OSError as e:
            print(f"Error: {filepath} : {e.strerror}")
    else:
        print(f"Configuration file {filepath} does not exist.")

deletecpuconfig()
cpu_config = CPUConfigurations(n_jobs=3, backend='loky')







