
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
video_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\DRA_raw.raw" 
dra_video = video_reader(file_name=video_path, type='binary', img_width=256, img_height=256,
                                   image_type=np.dtype('<u2'), s_frame=0, e_frame=-1) #Loading the video

original_shape= (2900, 256, 256)
dra_video = dra_video.reshape(original_shape)

Display(dra_video, time_delay=50)
n = np.shape(dra_video)[0]
frame_number = list(range(1, n))

# PSF = PSFsExtraction(video = dra_video ,flag_transform = True, flag_GUI = True)

<<<<<<< HEAD
df_PSFs = PSF.psf_detection(function='dog',  
                          min_sigma=5, max_sigma=8, sigma_ratio=1.5, threshold=8e-2,
                          overlap=0, mode='BOTH', frame_number = frame_number)
=======
#df_PSFs = PSF.psf_detection_preview(function='dog',  
 #                          min_sigma=5, max_sigma=8, sigma_ratio=1.5, threshold=8e-2,
   #                       overlap=0, mode='BOTH', frame_number = frame_number)
>>>>>>> 40e5873a1470dd98ed8d517ba413e483e9cb92c1
                        

# print(df_PSFs)
# save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif2.gif'

# display_psf = DisplayDataFramePSFsLocalization(dra_video, df_PSFs, 0.1, False , save_path)
# display_psf.cpu.parallel_active = False
# display_psf.run()



# display_psf.gif_genrator(save_path)
# display_psf.show_psf()
