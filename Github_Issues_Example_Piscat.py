# Github Issues Example for Piscat 

import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
import matplotlib.pyplot as plt

videoraw_path = r"C:\write\the\path\to\the\video.raw" 
videoraw_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\video_raw.raw" 

# load the raw video with piscat
video = video_reader(file_name=videoraw_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) 

video_pn, power = Normalization(video).power_normalized(inter_flag_parallel_active = True) 


 

