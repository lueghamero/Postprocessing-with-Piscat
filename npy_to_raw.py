# Convert from npy to raw 

import numpy as np
import PySimpleGUI as sg

# load the video
filename = sg.popup_get_file('Filename to play')
video_npy= np.load(filename)

# change it to raw name it and save
videoraw_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Raw_data\video_raw.raw" 
video_npy.tofile(videoraw_path)

