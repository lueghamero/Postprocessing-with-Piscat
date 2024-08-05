import ast
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
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
dra_video = instance_video.DifferentialAvg(opt_batch)
print(opt_batch)
Display(dra_video, time_delay=500)

