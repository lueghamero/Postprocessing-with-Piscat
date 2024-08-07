import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import tkinter as tk

def draw_figure(canvas_elem, figure):
    """Draw a Matplotlib figure on a Tkinter canvas."""
    for widget in canvas_elem.Widget.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=canvas_elem.Widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Define the layout for the main GUI
layout = [
    [sg.Text('Select a .npy file to display video frames:')],
    [sg.Input(key='-FILE-', enable_events=True), sg.FileBrowse(file_types=(('Numpy Files', '*.npy'),))],
    [sg.Button('Load Video'), sg.Button('Exit')],
    [sg.Canvas(key='-CANVAS-', size=(600, 600))],
    [sg.Button('Play'), sg.Button('Stop'), sg.Button('Next Frame'), sg.Button('Previous Frame')]
]

# Create the window
window = sg.Window('Video Player', layout, finalize=True)

# Create a Matplotlib figure and axes
fig, ax = plt.subplots()
canvas_elem = window['-CANVAS-']

video_data = None
playing = False

while True:
    event, values = window.read(timeout=100)  # Adjust timeout for responsiveness
    
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    
    if event == 'Load Video':
        # Load video data
        filename = values['-FILE-']
        if filename and os.path.isfile(filename):
            video_data = np.load(filename)
            vid_len = video_data.shape[0]
            frame_index = 0
            sg.popup(f'Loaded video with {vid_len} frames.')
            playing = False  # Stop any playing video when a new one is loaded

    if video_data is not None:
        # Update the frame
        frame = video_data[frame_index, :, :]
        ax.clear()
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame Number {frame_index + 1}')
        ax.axis('off')
        draw_figure(canvas_elem, fig)  # Draw the updated figure

        # Handle frame navigation
        if event == 'Next Frame':
            frame_index = (frame_index + 1) % vid_len  # Loop back to start if at end
            playing = False  # Stop automatic play when manually navigating frames
        elif event == 'Previous Frame':
            frame_index = (frame_index - 1) % vid_len  # Loop back to end if at start
            playing = False  # Stop automatic play when manually navigating frames
        elif event == 'Play':
            playing = True
        elif event == 'Stop':
            playing = False

        # Automatically advance frames if playing
        if playing:
            frame_index = (frame_index + 1) % vid_len

window.close()