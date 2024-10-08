from __future__ import print_function

import json
import os

import pandas as pd


class CPUConfig:
    def __init__(
        self,
        n_jobs=-1,
        backend="loky",
        verbose=10,
        parallel_active=True,
        threshold_for_parallel_run=None,
        flag_report=False,
        delete_file=False,
    ):
        """
        This class is copied from the class found in piscat.InputOutput cpu_configurations
        and a third definition for deleting the .json file was added in order to overwrite the 
        settings within the .json file.
        Also it is worth mentioning, that all the piscat functions are looking for that specific 
        .json file and load the configuration for parallel computing. 
        We also deleted the possibility to choose "multiprocessing" in the GUI, as it will not work,
        "loky" should just work fine though

        ---------------------------------------------------------------------------------------------
        This class generates a JSON file based on the parallel loop setting
        on the CPU that the user prefers.  This JSON was used by other
        functions and methods to set hyperparameters in a parallel loop.  For
        parallelization, PiSCAT used Joblib.

        | [1]. https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

        Parameters
        ----------
        n_jobs: int
            The maximum number of workers that can work at the same time.
            If -1, all CPU cores are available for use.

        backend: str
            Specify the implementation of the parallelization backend.
            The following backends are supported:

            * `“loky”`:
                It can induce some communication and Memory overhead when
                exchanging input and output data with the worker Python
                processes.

            * `“threading”`:
                It is a very low-overhead backend but it suffers from the
                Python Global Interpreter.  Lock if the called function relies
                a lot on Python objects. “threading” is mostly useful when the
                execution bottleneck is a compiled extension that explicitly
                releases the GIL (for instance a Cython loop wrapped in a “with
                nogil” block or an expensive call to a library such as NumPy).

        verbose: int, optional
            The verbosity level, if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.  The frequency of
            the messages increases with the verbosity level. If it more than
            10, all iterations are reported.

        parallel_active: bool
            Functions will run the parallel implementation if it is True.

        threshold_for_parallel_run: float
            It reserved for next generation of PiSCAT.

        flag_report: bool
            This flag is set if you need to see the values that will be used
            for CPU configuration.

        """
        try:
            self.read_cpu_setting(flag_report)
            self.delete_cpu_setting(delete_file)

        except FileNotFoundError:
            self.n_jobs = n_jobs
            self.backend = backend
            self.verbose = verbose
            self.parallel_active = parallel_active
            self.threshold_for_parallel_run = threshold_for_parallel_run
            self.delete_file = delete_file
            

            setting_dic = {
                "n_jobs": [self.n_jobs],
                "backend": [self.backend],
                "verbose": [self.verbose],
                "parallel_active": [self.parallel_active],
                "threshold_for_parallel_run": [self.threshold_for_parallel_run],
                "delete_file": [self.delete_file]
            }

            self.save_cpu_setting(setting_dic)

    def save_cpu_setting(self, setting_dic):
        name = "cpu_configurations.json"
        here = os.path.dirname(os.getcwd())
        subdir = "piscat_configuration"

        try:
            dr_mk = os.path.join(here, subdir)
            os.mkdir(dr_mk)
            print("Directory ", subdir, " Created ")
        except FileExistsError:
            print("Directory ", subdir, " already exists")

        filepath = os.path.join(here, subdir, name)
        df_configfile = pd.DataFrame(data=setting_dic)
        df_configfile.to_json(filepath)

    def read_cpu_setting(self, flag_report=False):
        """
        flag_report: bool
               Whether you need to see the values that will be used for CPU configuration.
        """
        subdir = "piscat_configuration"
        here = os.path.dirname(os.getcwd())
        filepath = os.path.join(here, subdir, "cpu_configurations.json")

        with open(filepath) as json_file:
            cpu_setting = json.load(json_file)

        self.n_jobs = cpu_setting["n_jobs"]["0"]
        self.backend = cpu_setting["backend"]["0"]
        self.verbose = cpu_setting["verbose"]["0"]
        self.parallel_active = cpu_setting["parallel_active"]["0"]
        self.threshold_for_parallel_run = cpu_setting["threshold_for_parallel_run"]["0"]

        if flag_report:
            print("PiSCAT's general parallel flag is set to {}".format(self.parallel_active))
            print("\nThe number of parallel jobs is set to {}".format(self.n_jobs))
            print("\nThe backend is set to {}".format(self.backend))
            print("\nThe verbose is set to {}".format(self.verbose))

    def delete_cpu_setting(self, delete_file=False):
        """
        Deletes the JSON configuration file if it exists.
        """
        if delete_file == True:
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
        else:
            pass

