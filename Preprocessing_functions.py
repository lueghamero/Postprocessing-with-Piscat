# Preprocessing_functions without class
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
import matplotlib.pyplot as plt
 
def Remove_Status_Line(video):
    status_ = read_status_line.StatusLine(video)  # Reading the status line
    video_sl, status_information = status_.find_status_line()  # Removing the status line
    return video_sl

def PowerNormalized(video, parallel= None):
    video_pn, power_fluctuation = Normalization(video).power_normalized(inter_flag_parallel_active=parallel) 
    # Plotting the power fluctuations 
    fig, ax = plt.subplots()
    # Plotting on the axes
    ax.plot(power_fluctuation, 'b', linewidth=1, markersize=0.5)
    ax.set_xlabel('Frame #', fontsize=18)
    ax.set_ylabel(r"$p / \bar p - 1$", fontsize=18)
    ax.set_title('Intensity fluctuations in the laser beam', fontsize=13)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Return the normalized video and the figure
    return video_pn #, fig
    
def DifferentialAvg(video, batch_size, mode_FPN):
    video_dr = DifferentialRollingAverage(video, batchSize=batch_size, mode_FPN=mode_FPN)
    video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
    return video_dra
    
def FindOptBatch(video, l_range): # Finding the perfect batch size 
    frame_number = len(video)
    noise_floor= NoiseFloor(video, list_range=l_range)
    # Optimal value for the batch size
    min_value = min(noise_floor.mean)
    min_index = noise_floor.mean.index(min_value)
    opt_batch = l_range[min_index]
    return opt_batch
    
def DarkFrameCorrection(video, dark_video, axis=None):
    # axis = 'None': the mean dark count could also be a good measure of the global offset due to dark counts.
    # axis = 0 (along the column), 1 (along the row)
    video_dfc = np.subtract(video, np.mean(dark_video, axis))
    return video_dfc
 
def RadialFiltering(video, rmin, rmax):
    # After PN and DRA
    rvt_ = RadialVarianceTransform(inter_flag_parallel_active=False)
    filtered_video = rvt_.rvt_video(video, rmin, rmax, kind="basic", highpass_size=None,
                            upsample=1, rweights=None, coarse_factor=1, coarse_mode='add',
                            pad_mode='constant')
    return filtered_video