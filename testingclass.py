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

# Remove the status line
result = instance_video.Remove_Status_Line()

# Power Normalization
video_sl_pn, fig = instance_video.PowerNormalized()  

# Show the plot if needed
#plt.show()

# Optimum Batch Size
l_range = list(range(30, 200, 30))
opt_batch = instance_video.FindOptBatch(l_range)

# Differential Rolling Average
mode_FPN='cpFPN'
dra_video = instance_video.DifferentialAvg(opt_batch, mode_FPN)
#print(opt_batch)
#Display(dra_video, time_delay=500)

PSF_1 = PSFsExtraction(video=dra_video)
PSFs = PSF_1.psf_detection(function='dog',
                            min_sigma=1.6, max_sigma=1.8, sigma_ratio=1.1, threshold=8e-4,
                            overlap=0, mode='BOTH')
DisplayDataFramePSFsLocalization(dra_video, PSFs, time_delay=0.1, GUI_progressbar=False)