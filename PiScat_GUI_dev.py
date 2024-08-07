import PySimpleGUI as sg
import numpy as np
import os
import sys
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
        [sg.Listbox(values=initial_file_list, size=(60, 10), key='-FILELIST-', enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
        [sg.Button("Read File"), sg.Button("Crop seleceted Video"), sg.Button("Cancel")],
        [sg.Button('Play'), sg.Button('Stop'), sg.Button('Next Frame'), sg.Button('Previous Frame')],
        [sg.Canvas(key='-CANVAS-', size=(600, 600))]
        
    ]

    # Create the window
    window = sg.Window('Folder and File Selector', vid_prep, finalize=True)
    
    return window

window = window_layout()
fig, ax = plt.subplots()
canvas_elem = window['-CANVAS-']

#%%#######################--------MAIN--------###########################
video_data = None
playing = False

# Event loop
while True:

    event, values = window.read(timeout=10)

    if event == sg.WINDOW_CLOSED:
        break

    if event == 'Cancel':
        break

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

    elif event == 'Read File':
        folder = values['-FOLDER-']
        selected_file = values['-FILELIST-'][0] if values['-FILELIST-'] else None
        if selected_file:
            full_path = os.path.join(folder, selected_file) # Read in the file
            if full_path and os.path.isfile(full_path):
                video_data = np.load(full_path)
                vid_len = video_data.shape[0]
                frame_index = 0
                sg.popup(f'Loaded video with {vid_len} frames.')
                playing = False   
        else:
            sg.popup(f'File could not be loaded: {full_path}')

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

    elif event == 'Crop seleceted Video':
        if 'full_path' in locals() and full_path:    
            cropper = vc(full_path)
            cropper.select_roi(frame_index=10)
            cropped_video_data = cropper.crop_video()
            cropper.save_cropped_video(cropped_video_data)
        else:
            sg.popup('Please load a video file first.')

window.close()