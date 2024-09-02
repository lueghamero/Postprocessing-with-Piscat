# tutorial 

import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

demo_video_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Tutorial4\Tutorial4_1\5nm_GNPs_128x128_uint16_3333fps_10Acc.raw"

video = video_reader(file_name=demo_video_path, type='binary', img_width=128, img_height=128,
                                    image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)#Loading the video
status_ = read_status_line.StatusLine(video[0:2000,:,:])#Reading the status line
video_remove_status, status_information  = status_.find_status_line()#Examining the status line & removing it


batchSize = 100
DRA = DifferentialRollingAverage(video=video_remove_status, batchSize=batchSize)
RVideo, _ = DRA.differential_rolling(FFT_flag=False)

# Display(RVideo, time_delay=50)
PSF_l = PSFsExtraction(video=RVideo)
PSF_l.cpu.parallel_active = False 
PSFs_dog = PSF_l.psf_detection(function='dog',
                            min_sigma=1.6, max_sigma=1.8, sigma_ratio=1.1, threshold=8e-4,
                            overlap=0, mode='BOTH')
print(PSFs_dog)

save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif2.gif'

DisplayDataFramePSFsLocalization(RVideo, PSFs_dog, 0.1, False, save_path).show_psf()