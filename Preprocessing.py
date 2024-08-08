import ast
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

from piscat_functions import PiscatFunctions

#load the video

video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/GNPs.npy'
video_file = np.load(video_path, allow_pickle=True)
video_file = video_file[200:600, :, :]
instance_video =  PiscatFunctions(video_file)

# Preprocessing 

# 1) Remove the status line
# video_sl = instance_video.Remove_Status_Line()

# 2) Dark Frame Correction 
# For the dark frame count you need to record the dark video. 
# Load the dark video

# filename = sg.popup_get_file('Filename to play') 
# video_dark = np.load(filename)
# video_dfc = instance_video.DarkFrameCorrection(video_dark)

# Comparing between Dark Frame Corrected and Not Corrected
# list_videos = [instance_video, video_dfc, instance_video - video_dfc]
# list_titles = ['Raw video', 'Video after \ndark frame correction', 'Difference']
# DisplaySubplot(list_videos, numRows=1, numColumns=3, step=1, median_filter_flag=False, color='gray')

# 3) Power Normalization
video_pn = instance_video.PowerNormalized()

# 4) Differential Rolling Average

# Find the Optimum Batch Size
# l_range = list(range(30, 200, 30))
# opt_batch = instance_video.FindOptBatch(l_range)
# print(opt_batch)

# choose between Fixed Pattern Noise Correction Methods: mFPN, cFPN, fFPN 
mode_FPN='mFPN'
video_pn_dra = instance_video.DifferentialAvg(15, mode_FPN, video= video_pn) 

# 5) Radial Variance Transform Filtering
# video_pn_dra_rf = instance_video.RadialFiltering(rmin=4, rmax=8, video= video_pn_dra )

Display(video_pn_dra, time_delay=200)

save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_GNPs.npy'
np.save(save_path, video_pn_dra)

