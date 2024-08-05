import PySimpleGUI as sg
import numpy as np
import os
import sys
from cropping_video import VideoCropper as vc

# Define the window's contents
layout = [
    [sg.Text("Please select a file:")],
    [sg.Input(key='-FILEPATH-', enable_events=True, readonly=True), sg.FileBrowse()],
    [sg.Multiline(size=(60, 10), key='-MLINE-', disabled=True)]
]

# Create the window
window = sg.Window('File Selector', layout)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == '-FILEPATH-':
        filepath = values['-FILEPATH-']
        window['-MLINE-'].update(filepath)

# Close the window
window.close()
 
"""
video_path = sg.popup_get_file('Filename to play')
output_path = os.path.dirname(video_path)

# Create a VideoCropper instance
cropper = vc(video_path)

# Select ROI
cropper.select_roi(frame_index=10)  # Select ROI on the 10th frame

# Crop the video based on the selected ROI
cropped_video_data = cropper.crop_video()

# Save the cropped video
cropper.save_cropped_video(cropped_video_data, output_path)

#Main Code for the iScat analysis.
"""
