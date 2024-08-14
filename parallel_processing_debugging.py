#parallel processing debugging

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

def main():
    # Load the DRA'd file
    video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_GNPs.npy'
    dra_video = np.load(video_path)
    dra_video = dra_video[10:40, :, :]
    # Display(dra_video, time_delay=300)

    n = np.shape(dra_video)[0]
    frame_number = [i for i in range(1, n)]
    
    PSF = PSFsExtraction(video=dra_video, flag_transform=False)

    # print(PSF)

    PSFs_RVT = PSF.psf_detection(function='RVT',
                                 min_radial=5, max_radial=6,  rvt_kind="basic",
                                 highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add",
                                 pad_mode="constant", threshold=1.5e-5)
    print(PSFs_RVT)
    
    save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif_rvt.gif'
    
    display_psf = DisplayDataFramePSFsLocalization(dra_video, PSFs_RVT, 0.1, False, save_path)
    display_psf.cpu.parallel_active = False
    print(display_psf)
    display_psf.run() 

if __name__ == '__main__':
    main()
