import PySimpleGUI as sg
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cropping_video import VideoCropper as vc
import tkinter as tk


#%%##################--------INITIAL--------########################

i=0
full_path = None

def draw_figure(canvas_elem, figure):
    """Draw a Matplotlib figure on a Tkinter canvas."""
    for widget in canvas_elem.Widget.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=canvas_elem.Widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

#create a folder for the config file, in which saves such as the filepath to the video is stored
CONFIG_DIR = 'config'
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.txt')


# Function to save the folder path to a config file
def save_folder_path(folder_path):
    with open(CONFIG_FILE, 'w') as file:
        file.write(folder_path)


# Function to load the folder path from a config file
def load_folder_path():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return file.read().strip()
    return ''


# Function to list files in a folder
def list_files_in_folder(folder, show_only_npy=False):
    try:
        files = os.listdir(folder)
        if show_only_npy:
            files = [f for f in files if f.lower().endswith('.npy')]
        return files
    except FileNotFoundError:
        return ["Folder not found."]
    except NotADirectoryError:
        return ["Selected path is not a folder."]
    except PermissionError:
        return ["Permission denied."]
    
def differential_view(video, frame_index, batchSize):
    """
    Basically the DRA function from piScat library, but modified
    """
    batch_1 = np.sum(video[0 : batchSize, :, :], axis=0)
    batch_2 = np.sum(video[batchSize : 2 * batchSize, :, :], axis=0)

    batch_1_ = np.divide(batch_1, batchSize)
    batch_2_ = np.divide(batch_2, batchSize)

    output_diff = batch_2_ - batch_1_

    batch_1 = (batch_1 - video[frame_index - 1, :, :] + video[batchSize + frame_index - 1, :, :])
    batch_2 = (batch_2 - video[batchSize + frame_index - 1, :, :]+ video[(2 * batchSize) + frame_index - 1, :, :]
                )
    batch_1_ = np.divide(batch_1, batchSize)
    batch_2_ = np.divide(batch_2, batchSize)

    output_diff = batch_2_ - batch_1_


    return output_diff


#Class for redirecting Terminal Output to GUI
class OutputRedirector:
    def __init__(self, window, key):
        self.window = window
        self.key = key

    def write(self, message):
        self.window.write_event_value(self.key, message)

    def flush(self):
        pass

# Load the last selected folder path
initial_folder = load_folder_path()
initial_show_only_npy = True
initial_file_list = list_files_in_folder(initial_folder, initial_show_only_npy) if initial_folder else []



#%%###################--------WINDOW_LAYOUT--------######################

def window_layout():

    vid_prep = [
        [sg.Text("Please select a folder:")],
        [sg.Input(key='-FOLDER-', enable_events=True, default_text=initial_folder), sg.FolderBrowse()],
        [sg.Checkbox('Show only .npy files', key='-SHOW_NPY-', default=initial_show_only_npy, enable_events=True)],
        [sg.Listbox(values=initial_file_list, size=(60, 10), key='-FILELIST-', auto_size_text=True, enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
        [sg.Button("Read Video"), sg.Button("Crop seleceted Video"), sg.Button("Read Darkframe Video"), sg.Button("Cancel")]
    ]


    vid_play = [
        [sg.Button('Play'), sg.Button('Stop'), sg.Button('Previous Frame'), sg.Button('Next Frame'), 
         sg.Checkbox("Differential View", key='-DIFF-', enable_events=True), sg.Text('Batch Size', size=(10,1)), 
         sg.Slider((1, 50), size=(10,5), default_value=1, key='-BATCH-', orientation='horizontal', enable_events=True)],
        [sg.Button('Reset',key='-RESET-')], 
        [sg.Canvas(key='-CANVAS-',pad=(0,0), size=(500, 500))]
    ]
        
    vid_layout = [
        [sg.Frame("Video Selection", vid_prep, expand_x=True, expand_y=True)],
        [sg.HorizontalLine()],
        [sg.Frame("Video Preview", vid_play)]
    ]

    piscat_prep = \
    [
        [sg.Multiline(size=(100,10), key='-OUTPUT-', autoscroll = True, background_color='black', text_color='white', reroute_stdout=True, reroute_stderr=True)]
    ]

    piscat_layout =     [
        [sg.Frame("PiScat", piscat_prep)]
    ]

    layout = [
        [sg.Column(vid_layout, expand_x=False, expand_y=False),sg.VerticalSeparator(),sg.Column(piscat_layout)]
    ]

    # Create the window
    window = sg.Window('iScat Postprocessing using PiScat', layout, size = (1600,900), resizable=True, finalize=True)
    
    return window

window = window_layout()

output_redirector = OutputRedirector(window, '-OUTPUT-')
sys.stdout = output_redirector
sys.stderr = output_redirector

fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(2.6,2.6))
canvas_elem = window['-CANVAS-']

#%%#######################--------MAIN--------###########################
video_data = None
playing = False

# Event loop
while True:

    event, values = window.read(timeout=5)
    batchSize = int(values['-BATCH-'])

    if event == sg.WINDOW_CLOSED:
        break

    if event == 'Cancel':
        break

    if event == '-OUTPUT-':
        window['-OUTPUT-'].print(values['-OUTPUT-'])

    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        show_only_npy = values['-SHOW_NPY-']
        file_list = list_files_in_folder(folder, show_only_npy)
        window['-FILELIST-'].update(file_list)
        save_folder_path(folder)  # Save the selected folder path

    elif event == '-SHOW_NPY-': # Only show .npy files
        folder = values['-FOLDER-']
        show_only_npy = values['-SHOW_NPY-']
        if folder:
            file_list = list_files_in_folder(folder, show_only_npy)
            window['-FILELIST-'].update(file_list)

    elif event == '-FILELIST-':
        # Update selected file path, no need to change the listbox
        pass

    elif event == 'Read Video':
        folder = values['-FOLDER-']
        selected_file = values['-FILELIST-'][0] if values['-FILELIST-'] else None
        if selected_file:
            full_path = os.path.join(folder, selected_file) # Read in the file
            if full_path and os.path.isfile(full_path):
                video_data = np.load(full_path)
                vid_len = video_data.shape[0]
                frame_index = 0
                sg.popup(f'Loaded video has {vid_len} frames.')
                playing = False   
        else:
            sg.popup(f'File could not be loaded: {full_path}')

    if video_data is not None:
        # Update the frame
        
        if event == '-RESET-':
            frame_index = 0

        if values['-DIFF-'] == True:
            frame = differential_view(video_data, frame_index, batchSize)
        else:
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

    elif event == 'Crop seleceted Video':
        if 'full_path' in locals() and full_path:    
            cropper = vc(full_path)
            cropper.select_roi(frame_index=10)
            cropped_video_data = cropper.crop_video()
            cropper.save_cropped_video(cropped_video_data)
        else:
            sg.popup('Please load a video file first.')

window.close()

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
