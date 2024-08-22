
import numpy as np

from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

from Preprocessing_functions import *

# path to the raw video
videoraw_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\video_raw.raw" 

# load the raw video with piscat
video = video_reader(file_name=videoraw_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) #Loading the video
print(video.shape[0])
# print(video_file.shape)

# Display(video_file, time_delay=50)

# Preprocessing 

# 1) Remove the status line
# vdeo_sl = Remove_Status_Line(video)

# 2) Dark Frame Correction 
# For the dark frame count you need to record the dark video. 

# path to the raw video
darkframe_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\dark_frame_raw.raw" 

# load the raw video with piscat
dark_video = video_reader(file_name=darkframe_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) #Loading the video

video_dfc = DarkFrameCorrection(video, dark_video, axis=None)
# Display(video_dfc, time_delay=10)
# Comparing between Dark Frame Corrected and Not Corrected
# list_videos = [video, video_dfc, video - video_dfc]
# list_titles = ['Raw video', 'Video after \ndark frame correction', 'Difference']
# DisplaySubplot(list_videos, numRows=1, numColumns=3, step=1, median_filter_flag=False, color='gray')

# 3) Power Normalization
video_pn, power = Normalization(video_dfc).power_normalized() 
# video_pn = PowerNormalized(video_dfc, parallel= False) 

# Display(video_pn, time_delay=10)
# 4) Differential Rolling Average

# Find the Optimum Batch Size
# l_range = list(range(30, 200, 30))
# opt_batch = instance_video.FindOptBatch(l_range)
# print(opt_batch)

# choose between Fixed Pattern Noise Correction Methods: mFPN, cFPN, fFPN 
mode_FPN='mFPN'
video_pn_dra = DifferentialAvg(video_pn[0:3000,:,:], 50, mode_FPN) 
# Display(video_pn_dra, time_delay=50)
# 5) Radial Variance Transform Filtering
# video_pn_dra_rf = instance_video.RadialFiltering(rmin=4, rmax=8, video= video_pn_dra )

# Saving the DRA video
# Display(video_pn_dra, time_delay=200)
# DRAd_raw_save = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\DRA_raw.raw" 
# video_pn_dra.tofile(DRAd_raw_save)

# 6) PSF detection

n = np.shape(video_pn_dra)[0]
frame_number = list(range(1, n))
# PSF = PSFsExtraction(video = video_pn_dra, flag_transform = True, flag_GUI = True)
# PSF.cpu.parallel_active =False
# df_PSFs = PSF.psf_detection_preview(function='dog',  
 #                         min_sigma=5, max_sigma=8, sigma_ratio=1.5, threshold=8e-2,
   #                      overlap=0, mode='BOTH', frame_number = frame_number)
# print(df_PSFs)