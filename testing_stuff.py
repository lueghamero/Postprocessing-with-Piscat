
import numpy as np
import cv2 as cv
import PySimpleGUI as sg
import os
import sys
import pylab
from piscat.InputOutput import reading_videos
from piscat.Preproccessing import Normalization
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




"""
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
"""


"""
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


