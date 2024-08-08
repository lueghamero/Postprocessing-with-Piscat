import ast
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt

from piscat_functions import PiscatFunctions
import dill 

# Localization of particles after DRA

# Load the DRA'd file
video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_GNPs.npy'
dra_video = np.load(video_path)
dra_video = dra_video[10:60, :, :]
# Display(dra_video, time_delay=300)

n = np.shape(dra_video)[0]
frame_number = [i for i in range(1, n)]

# PSF = PSFsExtraction(video = dra_video ,flag_transform = True, flag_GUI = True)

# df_PSFs = PSF.psf_detection_preview(function='dog',  
#                            min_sigma=2, max_sigma=5, sigma_ratio=1.5, threshold=8e-3,
#                             overlap=0, mode='BOTH', frame_number = frame_number)
#print(df_PSFs)

PSF= particle_localization.PSFsExtraction(video=dra_video, flag_transform=False)
PSF.cpu.parallel_active = False
PSFs_RVT = PSF.psf_detection(function='RVT',
                            min_radial=2, max_radial=3,  rvt_kind="basic",
                            highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add",
                            pad_mode="constant", threshold=1.5e-7)


save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif1.gif'

display_psf = DisplayDataFramePSFsLocalization(dra_video, PSFs_RVT, 0.1, False , save_path)
# display_psf.cpu.parallel_active = False
# display_psf.run()

# print(dill.detect.baditems(display_psf))

# display_psf.gif_genrator(save_path)
display_psf.show_psf()

# display_psf.gif_genrator(save_path)
