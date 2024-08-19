
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

from piscat_functions import PiscatFunctions

# load the video
video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/BsaBioStrep.npy'
video = np.load(video_path)

# change it to raw and save
videoraw_path = "/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/video.raw" 
video.tofile(videoraw_path)

# load the raw video with piscat
video_file = video_reader(file_name=videoraw_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) #Loading the video

instance_video =  PiscatFunctions(video_file)
# print(video_file.shape)
# Display(video_file, time_delay=500)
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

# video_pn = instance_video.PowerNormalized()
# Display(video_pn, time_delay=500)
# 4) Differential Rolling Average

# Find the Optimum Batch Size
# l_range = list(range(30, 200, 30))
# opt_batch = instance_video.FindOptBatch(l_range)
# print(opt_batch)

# choose between Fixed Pattern Noise Correction Methods: mFPN, cFPN, fFPN 
mode_FPN='cFPN'
video_pn_dra = instance_video.DifferentialAvg(50, mode_FPN) 
Display(video_pn_dra, time_delay=50)
# 5) Radial Variance Transform Filtering
# video_pn_dra_rf = instance_video.RadialFiltering(rmin=4, rmax=8, video= video_pn_dra )

# Display(video_pn_dra, time_delay=200)


DRAd_raw_save = "/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_bio.raw" 
video_pn_dra.tofile(DRAd_raw_save)


#print(video_pn_dra)
#print(video_pn_dra.shape)