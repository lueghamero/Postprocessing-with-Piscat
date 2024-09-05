import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import time
from PSF_localization_preview import *

# Assume PSFsExtractionPreview is your existing class, instantiated here:
psf_extractor = PSFsExtractionPreview(video=None, min_sigma=1, max_sigma=3, threshold=0.02)

# Sample video (replace this with actual video frames)
video = np.random.random((100, 256, 256))  # 100 frames of 256x256 images

# GUI layout
layout = [
    [sg.Canvas(key='-CANVAS-')],
    [sg.Text('Min Sigma'), sg.Slider(range=(0.1, 10), resolution=0.1, default_value=1, key='-MIN_SIGMA-', orientation='h')],
    [sg.Text('Max Sigma'), sg.Slider(range=(0.1, 10), resolution=0.1, default_value=3, key='-MAX_SIGMA-', orientation='h')],
    [sg.Button('Play'), sg.Button('Stop')],
]
window = sg.Window('Video Playback with PSF Detection', layout)

# Initialize Matplotlib figure
fig, ax = plt.subplots()
canvas_elem = window['-CANVAS-']

# Initialize video playback variables
playing = False
frame_index = 0
frame_rate = 30  # frames per second

# Function to update the plot
def update_plot(frame, ax, psf_data=None):
    ax.clear()
    ax.imshow(frame, cmap='gray')
    
    if psf_data is not None:
        # Add red circles to detected PSFs
        psf_extractor.add_circles_to_ax(ax, psf_data)

    ax.set_title(f'Frame {frame_index + 1}')
    ax.axis('off')

    # Draw the figure on the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

while True:
    event, values = window.read(timeout=10)

    if event == sg.WIN_CLOSED:
        break

    # Get the current sigma values from sliders
    min_sigma = values['-MIN_SIGMA-']
    max_sigma = values['-MAX_SIGMA-']
    
    # Update the class's sigma values dynamically
    psf_extractor.min_sigma = min_sigma
    psf_extractor.max_sigma = max_sigma

    if event == 'Play':
        playing = True

    if event == 'Stop':
        playing = False

    if playing:
        # Load the current frame
        frame = video[frame_index]

        # Process the current frame with the PSF detection class
        psf_data = psf_extractor.psf_detection(frame, frame_index, function='dog')

        # Update the plot with the current frame and detected PSFs
        update_plot(frame, ax, psf_data)

        # Increment frame index, looping back to the start if necessary
        frame_index = (frame_index + 1) % video.shape[0]

        # Control playback speed
        time.sleep(1 / frame_rate)

window.close()


