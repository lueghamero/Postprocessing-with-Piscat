# Exploring piscat functions

import ast
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
import matplotlib.pyplot as plt

# Step 1: Path to video and text file
txt_file_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/BsaBioStrep.txt'
video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/BsaBioStrep.npy'

# load video and the text file
video_file = np.load(video_path, allow_pickle=True)
text_file = open(txt_file_path, 'r').read()

# Step 2: Convert the string content to a dictionary
data_dict = ast.literal_eval(text_file)

# Step 3: Extract the a key value, e.g., photon_fps
photon_fps = data_dict.get('photon_fps')

print(f"Type of video_frames: {type(video_file)}")
print(f"Number of frames: {len(video_file)}")
if len(video_file) > 0:
    print(f"Shape of a single frame: {video_file[0].shape}")

def Remove_Status_Line(video):
    status_ = read_status_line.StatusLine(video_file) # Reading the status line
    video_sl, status_information  = status_.find_status_line() # Removing the status line
    return video_sl

def PowerNormalized(video):
    video_pn, power_fluctuation = Normalization(video).power_normalized()
    return video_pn

def DifferentialAvg(video, batch_size):
    video_dr = DifferentialRollingAverage(video=video_sl_pn, batchSize=batch_size, mode_FPN='mFPN')
    video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
    return video_dra


video_sl = Remove_Status_Line(video_file) # Removing the status line
video_sl_pn = PowerNormalized(video_sl) # Power Normalization
video_sl_pn_dra = DifferentialAvg(video_sl_pn, 1) # Differential Average Rolling


#Display(video_sl_pn_dra,time_delay=500) 

# Finding the perfect batch size 
frame_number= len(video_file)
l_range = list(range(30, 200, 30))

# Noise floor calculation with different batch sizes

noise_floor_sl_pn_dra = NoiseFloor(video_sl_pn_dra, list_range=l_range)

# Optimal value for the batch size
min_value = min(noise_floor_sl_pn_dra.mean)
min_index = noise_floor_sl_pn_dra.mean.index(min_value)
opt_batch = l_range[min_index]
print(opt_batch) 

# plt.plot(l_range, noise_floor_sl_pn.mean, label='PN')
""" plt.plot(l_range, noise_floor_sl_pn_dra.mean, label='PN+DRA')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel("Batch size", fontsize=18)
plt.ylabel("Noise floor", fontsize=18)
plt.legend()
plt.show()
 """

