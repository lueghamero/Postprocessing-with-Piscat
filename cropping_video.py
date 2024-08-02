import numpy as np
import PySimpleGUI as sg
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


filename = sg.popup_get_file('Filename to play')

if filename is None:
    exit()
#My version of extracting folder in which data is stored and name of data:
filename_folder = os.path.dirname(filename)
filename_measurement = os.path.splitext(os.path.basename(filename))[0]

video_data = np.load(filename)
#video_data = np.transpose(video_data, (1, 2, 0))

# Global variable to store the ROI
roi = None

# Function to handle ROI selection
def onselect(eclick, erelease):
    global roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    plt.close()

# Choose a frame to display (e.g., the 10th frame)
frame_index = 10
frame = video_data[frame_index]

# Display the frame and select ROI
fig, ax = plt.subplots()
ax.imshow(frame, cmap='gray')
ax.set_title("Select ROI and close the window")
rect_selector = RectangleSelector(ax, onselect, 
                                   button=[1], minspanx=5, minspany=5, 
                                   spancoords='pixels', interactive=True)
plt.show()

# Validate the ROI
if roi is None:
    raise ValueError("ROI must be selected before cropping.")

x, y, w, h = roi
crop_size = (w, h)

# Create an array to store cropped frames
num_frames = video_data.shape[0]
cropped_video_data = np.zeros((num_frames, h, w), dtype=video_data.dtype)

# Crop each frame
for frame_index in range(num_frames):
    frame = video_data[frame_index]
    cropped_frame = frame[y:y+h, x:x+w]
    cropped_video_data[frame_index] = cropped_frame

# Save the cropped video to a .npy file
cropped_video_path = filename_folder
np.save(cropped_video_path, cropped_video_data)
print(f"Cropped video saved to {cropped_video_path}")