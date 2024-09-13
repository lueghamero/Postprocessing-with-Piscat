
import numpy as np
import cv2 as cv
import PySimpleGUI as sg
import os
import sys
import pylab
from piscat.InputOutput import reading_videos
from piscat.Visualization import * 
from piscat.Preproccessing import Normalization
from piscat.BackgroundCorrection import NoiseFloor
from piscat.BackgroundCorrection import DifferentialRollingAverage
from piscat.Localization import *
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
from PSF_localization_preview_copy import *


#filename = sg.popup_get_file('Filename to play')
#filename = r"C:\Users\Emanuel\Desktop\Masterarbeit\2024_02_27_data\12_59_32_Sample1_Refrence_Air.npy"
filename = r"C:\Users\Dante\Desktop\Processed_iScat_Vids/15_35_12_Au_3mul_15042024_DRA_filtered.npy"
#if filename is None:
#    exit()
#My version of extracting folder in which data is stored and name of data:
filename_folder = os.path.dirname(filename)
filename_measurement = os.path.splitext(os.path.basename(filename))[0]

video_data = np.load(filename, allow_pickle=True)
video_dra = video_data[:,:,:]
#video_data = np.transpose(video_data, (1, 2, 0))

#video_pn, power_fluctuation = Normalization(video=video_data).power_normalized()

#video_dr = DifferentialRollingAverage(video=video_pn, batchSize=30, mode_FPN='fFPN')
#video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)

# Assume 'video' is your video array and has shape (num_frames, height, width)
n = np.shape(video_dra)[0]

frame_number = list(range(1, n))
PSFs = PSFsExtraction(video_dra)


# Detect PSFs in the frame
psf_positions = PSFs.psf_detection( function='dog', min_sigma=2.5, max_sigma=6, sigma_ratio=1.1, threshold= 0.0042, overlap = 0)
#psf_positions_filtered = SpatialFilter().remove_side_lobes_artifact(psf_positions)
#psf_positions_filtered = SpatialFilter().dense_PSFs(psf_positions_filtered)

print(psf_positions)
#print(psf_positions_filtered)


#isplay_psf_loaded = DisplayDataFramePSFsLocalization( video_dra, psf_positions,  0.1, False)
#display_psf_loaded.show_psf(display_history=False)




#PSFshow = PSF.psf_detection_preview(function='dog', min_sigma=1, max_sigma=8, sigma_ratio=1.5, threshold=0.00008, overlap=0, mode='BOTH', frame_number=100, IntSlider_width='400px')



