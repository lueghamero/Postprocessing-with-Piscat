from ParallelProcessingConfig import CPUConfig
from piscat.InputOutput import CPUConfigurations
import json
import os

import pandas as pd



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
cpu_config_new = CPUConfigurations(n_jobs=5)
print(cpu_config_new.n_jobs)






