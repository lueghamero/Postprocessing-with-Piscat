
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

filename = sg.popup_get_file('Filename to play')
#filename = r"C:\Users\Emanuel\Desktop\Masterarbeit\2024_02_27_data\12_59_32_Sample1_Refrence_Air.npy"

if filename is None:
    exit()
#My version of extracting folder in which data is stored and name of data:
filename_folder = os.path.dirname(filename)
filename_measurement = os.path.splitext(os.path.basename(filename))[0]

video_data = np.load(filename)
#video_data = np.transpose(video_data, (1, 2, 0))

video_pn, power_fluctuation = Normalization(video=video_data).power_normalized()

video_dr = DifferentialRollingAverage(video=video_pn, batchSize=30, mode_FPN='mFPN')
video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)

PSF = PSFsExtraction(video = video_dra ,flag_transform = True, flag_GUI = True)

df_PSFs = PSF.psf_detection(function='dog',  
                          min_sigma=5, max_sigma=8, sigma_ratio=1.5, threshold=8e-2,
                          overlap=0, mode='BOTH')
                        

print(df_PSFs)









"""
# Ensure the pixel values are within the correct range
if video_data.dtype != np.uint8 or video_data.min() < 0 or video_data.max() > 255:
    # Normalize the pixel values to the range 0-255 and convert to uint8
    video_data = ((video_data - video_data.min()) / (video_data.max() - video_data.min()) * 255).astype(np.uint8)

video_pn, power_fluctuation = Normalization(video=video_data).power_normalized()

l_range = list(range(30, 300, 30))
noise_floor_DRA_pn = NoiseFloor(video_pn, list_range=l_range)

min_value = min(noise_floor_DRA_pn.mean)
min_index = noise_floor_DRA_pn.mean.index(min_value)
opt_batch = l_range[min_index]

def DifferentialAvg(video, batch_size):
    video_dr = DifferentialRollingAverage(video=video, batchSize=batch_size, mode_FPN='mFPN')
    video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
    return video_dra

processed_vid = DifferentialAvg(video_pn, 10)

Display(processed_vid,time_delay=500) 

# Print the shape of the array to understand its structure
print("Shape of the video data:", processed_vid.shape)



# Release the video window
cv2.destroyAllWindows()

# Calculate the center of the frame
center_x, center_y = frame_width // 2, frame_height // 2
crop_size = 128

# Calculate cropping box coordinates
start_x = center_x - crop_size // 2
end_x = start_x + crop_size
start_y = center_y - crop_size // 2
end_y = start_y + crop_size

# Ensure the coordinates are within the frame dimensions
start_x = max(0, start_x)
end_x = min(frame_width, end_x)
start_y = max(0, start_y)
end_y = min(frame_height, end_y)

# Iterate over each frame and display it
for frame_index in range(num_frames):
    frame = video_data[frame_index]
    cropped_frame = frame[start_y:end_y, start_x:end_x]
    
    # Check if the frame is grayscale or color
    if cropped_frame.ndim == 2:  # Grayscale
        cv2.imshow('Video', cropped_frame)
    elif cropped_frame.ndim == 3:  # Color
        cv2.imshow('Video', cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"Unexpected frame dimensions: {cropped_frame.shape}")

    # Wait for 25 ms between frames (40 FPS)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the video window
cv2.destroyAllWindows()




df_video = reading_videos.DirectoryType(filename, type_file='raw').return_df()
paths = df_video['Directory'].tolist()
video_names = df_video['File'].tolist()

video = np.load(filename)
video = np.transpose(video, (1,2,0))
vid_norm = np.sum(video, (0,1))/(np.shape(video)[0]*np.shape(video)[1])
vid = video/vid_norm

video_pn, power_fluctuation = Normalization(video=video).power_normalized()


plt.plot(power_fluctuation, 'b', linewidth=1, markersize=0.5)
plt.xlabel('Frame #', fontsize=18)
plt.ylabel(r"$p / \bar p - 1$", fontsize=18)
plt.title('Intensity fluctuations in the laser beam', fontsize=13)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
"""


