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

video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/BsaBioStrep.npy'
video_file = np.load(video_path, allow_pickle=True)

instance_video =  PiscatFunctions(video_file)

# 1) Remove the status line
video_sl = instance_video.Remove_Status_Line()

# 2) Power Normalization
video_sl_pn = instance_video.PowerNormalized(video_sl)

# 3) Dark Frame Correction
video_sl_dfc = instance_video.DarkFrameCorrection(video_sl, axis=0)
# Display(video_sl_pn_dc, time_delay=500)

# Comparing between Dark Frame Corrected and Not Corrected
# list_videos = [video_sl, video_sl_dfc, video_sl - video_sl_dfc]
# list_titles = ['Raw video', 'Video after \ndark frame correction', 'Difference']
# DisplaySubplot(list_videos, numRows=1, numColumns=3, step=1, median_filter_flag=False, color='gray')

# Show the plot if needed
#plt.show()

# Optimum Batch Size
l_range = list(range(30, 200, 30))
opt_batch = instance_video.FindOptBatch(l_range)

# Differential Rolling Average
mode_FPN='cpFPN'
dra_video = instance_video.DifferentialAvg(opt_batch, mode_FPN)
# print(opt_batch)
# Display(dra_video, time_delay=500)

n = np.shape(dra_video)[0]
frame_number = [i for i in range(1, n)]

PSF = PSFsExtraction(video = dra_video ,flag_transform = True, flag_GUI = True)

df_PSFs = PSF.psf_detection_preview(function='dog',  
                            min_sigma=1.5, max_sigma=3.5, sigma_ratio=1.5, threshold=2.002e-3,
                            overlap=0, mode='BOTH', frame_number = frame_number)
print(df_PSFs)

display_psf= DisplayDataFramePSFsLocalization(dra_video, df_PSFs, time_delay=0.1, GUI_progressbar=False)
display_psf.run()