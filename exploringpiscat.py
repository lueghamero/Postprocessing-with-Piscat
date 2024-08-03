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
    # Plotting the power fluctuations 
    fig, ax = plt.subplots()
    # Plotting on the axes
    ax.plot(power_fluctuation, 'b', linewidth=1, markersize=0.5)
    ax.set_xlabel('Frame #', fontsize=18)
    ax.set_ylabel(r"$p / \bar p - 1$", fontsize=18)
    ax.set_title('Intensity fluctuations in the laser beam', fontsize=13)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Return the normalized video and the figure
    return video_pn, fig

def DifferentialAvg(video, batch_size):
    video_dr = DifferentialRollingAverage(video=video_sl_pn, batchSize=batch_size, mode_FPN='mFPN')
    video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
    return video_dra

def FindOptBatch(video_file,l_range): # Finding the perfect batch size 
    frame_number = len(video_file)
    noise_floor= NoiseFloor(video_file, list_range=l_range)
    # Optimal value for the batch size
    min_value = min(noise_floor.mean)
    min_index = noise_floor.mean.index(min_value)
    opt_batch = l_range[min_index]
    return opt_batch


video_sl = Remove_Status_Line(video_file) # Removing the status line
video_sl_pn, fig = PowerNormalized(video_sl)  # Power Normalization

# Show the plot if needed
plt.show()



