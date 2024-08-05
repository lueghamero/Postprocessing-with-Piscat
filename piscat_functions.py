# piscat_functions.py
import ast
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
import matplotlib.pyplot as plt

class PiscatFunctions: 
    
    def __init__(self, video):
        self.video = video

    def Remove_Status_Line(self):
        status_ = read_status_line.StatusLine(self.video)  # Reading the status line
        video_sl, status_information = status_.find_status_line()  # Removing the status line
        return video_sl

    def PowerNormalized(self):

        video_pn, power_fluctuation = Normalization(self.video).power_normalized() 

        # Plotting the power fluctuations 
        fig, ax = plt.subplots()
        # Plotting on the axes
        ax.plot(power_fluctuation, 'b', linewidth=1, markersize=0.5)
        ax.set_xlabel('Frame #', fontsize=18)
        ax.set_ylabel(r"$p / \bar p - 1$", fontsize=18)
        ax.set_title('Intensity fluctuations in the laser beam', fontsize=13)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # Return the normalized video and the figure
        return video_pn, fig
    
    def DifferentialAvg(self, batch_size, mode_FPN):
        video_dr = DifferentialRollingAverage(video=self.video, batchSize=batch_size, mode_FPN=mode_FPN)
        video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
        return video_dra
    
    def FindOptBatch(self, l_range): # Finding the perfect batch size 
        frame_number = len(self.video)
        noise_floor= NoiseFloor(self.video, list_range=l_range)
        # Optimal value for the batch size
        min_value = min(noise_floor.mean)
        min_index = noise_floor.mean.index(min_value)
        opt_batch = l_range[min_index]
        return opt_batch