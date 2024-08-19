
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

from piscat_functions import PiscatFunctions


# Localization of particles after DRA

# Load the DRA'd file 
video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_Bio.raw'
dra_video = video_reader(file_name=video_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) #Loading the video
Display(dra_video, time_delay=200)
print(dra_video.shape)
n = np.shape(dra_video)[0]
frame_number = list(range(1, n))

PSF = PSFsExtraction(video = dra_video ,flag_transform = True, flag_GUI = True)

df_PSFs = PSF.psf_detection_preview(function='dog',  
                          min_sigma=5, max_sigma=8, sigma_ratio=1.5, threshold=8e-2,
                          overlap=0, mode='BOTH', frame_number = frame_number)
                        

print(df_PSFs)
# save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif2.gif'

# display_psf = DisplayDataFramePSFsLocalization(dra_video, df_PSFs, 0.1, False , save_path)
# display_psf.cpu.parallel_active = False
# display_psf.run()



# display_psf.gif_genrator(save_path)
# display_psf.show_psf()
