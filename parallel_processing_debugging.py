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

# Localization of particles after DRA

def main():
    # Load the DRA'd file
    video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_GNPs.npy'
    dra_video = np.load(video_path)
    dra_video = dra_video[10:50, :, :]
    # Display(dra_video, time_delay=300)

    n = np.shape(dra_video)[0]
    frame_number = range(1, n)
    
    PSF = particle_localization.PSFsExtraction(video=dra_video, flag_transform=False)

    # print(PSF)

    # PSFs_RVT = PSF.psf_detection(function='RVT',
    #                             min_radial=5, max_radial=6,  rvt_kind="basic",
    #                             highpass_size=None, upsample=1, rweights=None, coarse_factor=1, coarse_mode="add",
    #                             pad_mode="constant", threshold=3e-5)
    # print(PSFs_RVT)

    PSFs_dog = PSF.psf_detection(function='dog',
                            min_sigma=1.6, max_sigma=1.8, sigma_ratio=1.1, threshold=8e-4,
                            overlap=0, mode='BOTH')

    save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif_rvt.gif'
    
    display_psf = DisplayDataFramePSFsLocalization(dra_video, PSFs_dog, 0.1, False, save_path)
    display_psf.cpu.parallel_active = False
    print(display_psf)
    display_psf.run()
    # display_psf.show_psf() 

if __name__ == '__main__':
    main()
