import PySimpleGUI as sg
import numpy as np
import os
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cropping_video import VideoCropper as vc
import tkinter as tk
import skimage

from piscat.Visualization import * 
from piscat.Preproccessing import *
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *

#%%##################--------FUNCTIONS--------########################


# creates the figure canvas for the GUI
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
def save_file_paths(file_path1,file_path2):
    with open(CONFIG_FILE, 'w') as file:
        file.write(f"{file_path1}\n{file_path2}")


# Function to load the folder path from a config file
def load_file_paths():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            paths = file.read().strip().split('\n')
            return paths[0], paths[1] if len(paths) > 1 else ''
    return '', ''


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

# Function for the differential view with adjustable batch sizes
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

def delete_CPU_config():

    subdir = "piscat_configuration"
    here = os.path.abspath(os.path.join(os.getcwd(), '..'))
    filepath = os.path.join(here, subdir, "cpu_configurations.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Configuration file {filepath} deleted successfully.")
        except OSError as e:
            print(f"Error: {filepath} : {e.strerror}")
    else:
        print(f"Configuration file {filepath} does not exist.")



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
folder, full_path_dark = load_file_paths()
initial_show_only_npy = True
initial_file_list1 = list_files_in_folder(folder, initial_show_only_npy) if folder else []
dark_frame_video = np.load(full_path_dark) if full_path_dark else []

cpu_settings = CPUConfigurations()
n_jobs=cpu_settings.n_jobs
backend=cpu_settings.backend
verbose=cpu_settings.verbose,
parallel_active=cpu_settings.parallel_active


#%%###################--------WINDOW_LAYOUT--------######################

def window_layout():

    menu_def = [['&Options', ['CPU Configuration']]]

    vid_prep = [
        [sg.Text("Please select a folder:")],
        [sg.Input(key='-FOLDER-', enable_events=True, default_text=folder), sg.FolderBrowse()],
        [sg.Checkbox('Show only .npy files', key='-SHOW_NPY-', default=initial_show_only_npy, enable_events=True)],
        [sg.Listbox(values=initial_file_list1, size=(60, 10), key='-FILELIST-', auto_size_text=True, enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
        [sg.Button("Read Video"), sg.Button("Crop seleceted Video"), sg.Button("Select as Darkframe Video"),
          sg.OptionMenu(["Enable Video Player: NO", "Enable Video Player: YES"], default_value = "Enable Video Player: NO", key = '-PLAY OPTION-', background_color='lightgrey', text_color='black')]
    ]

    vid_play = [
        [sg.Button('Play'), sg.Button('Stop'), sg.Button('Previous Frame'), 
         sg.Slider(range=(1,5000), size=(20,10), orientation='h', key='-FRNR-', enable_events=True, disabled=True), sg.Button('Next Frame'), sg.Button('Reset',key='-RESET-')], 
        [sg.Checkbox("Differential View", key='-DIFF-', enable_events=True), sg.Text('Batch Size:', size=(10,1)), 
         sg.Slider((1, 160), size=(10,10), default_value=1, key='-BATCH-', orientation='horizontal', enable_events=True),sg.Checkbox("Show filtered Video", key='-FILTVID-', enable_events=True)], 
        [sg.Push(), sg.Canvas(key='-CANVAS-',pad=(0,0)), sg.Push()]
    ]
    
    vid_layout = [
        [sg.Frame("Video Selection", vid_prep, expand_x=True, expand_y=True)],
        [sg.HorizontalLine()],
        [sg.Frame("Video Preview", vid_play, expand_x=True, expand_y=True)]
    ]

    terminal_prep = [  
        [sg.Multiline(size=(150,15), key='-OUTPUT-',expand_x=True, autoscroll = True, background_color='black',
                       text_color='white', reroute_stdout=True, reroute_stderr=True)]
    ]

    power_normalization_df_correction = [
            [sg.Checkbox("Power Normalisation", key='-PN-', enable_events=True)],
            [sg.Checkbox("Darkframe Correction", key='-DFC-', enable_events=True)],
            [sg.Button('Preprocess Video')]

    ]

    DRA = [
        [sg.Column([[sg.Text('FPN correction mode')],[sg.Combo(['DRA_PN', 'cpFPNc', 'mFPNc', 'wFPNc', 'fFPNc'], key='-FPNc-', default_value='DRA_PN', size=(10), enable_events=True, readonly=True)],[sg.Button('DRA Filtering')]],vertical_alignment='top'),
         sg.Column([[sg.Text('Batch Size')], [sg.Input(default_text='1', size=(10), key='-BATCH_IN-', enable_events=True)], [sg.Button('Find Optimal Batch Size')]],vertical_alignment='top')]
    ]

    piscat_preprocessing = [
            [sg.Column(power_normalization_df_correction, vertical_alignment='top'), sg.VerticalSeparator(), sg.Column(DRA, vertical_alignment='top')],
            [sg.Canvas(key='-PNCANV-', pad=(0,0), size=(1000,520))]    
    ]

    piscat_particle_detection = [

    ]

    tab_layout = [
        [sg.TabGroup([[sg.Tab("Preprocessing", piscat_preprocessing), sg.Tab("Particle Detection", piscat_particle_detection)]])]
    ]

    piscat_layout =     [
        [sg.Frame("Piscat", tab_layout, expand_x=True, expand_y=True)],
        [sg.HorizontalSeparator()],
        [sg.Frame("Terminal", terminal_prep, expand_x=True)]
    ]

    layout = [
        [sg.Menu(menu_def)],
        [sg.Column(vid_layout, vertical_alignment='top'), sg.VerticalSeparator(), 
         sg.Column(piscat_layout, vertical_alignment='top', expand_x=True, expand_y=True)]
    ]

    # Create the window
    window = sg.Window('iScat Postprocessing using PiScat', layout, size = (1600,900), resizable=True, finalize=True)
    
    return window

def CPU_window_layout():

    left_column = [
        [sg.Text('Number of Cores:')],
        [sg.Text('Verbosity Level:')],
        [sg.Text('Parallel Method:')]
    ]

    right_column = [
        [sg.Input(default_text='-1', size=(20), key='-N_JOBS-')],
        [sg.Input(default_text='10', size=(20), key='-VERBOS-')],
        [sg.Combo(['loky', 'threading'], key='-BACKEND-', default_value='loky', size=(20), enable_events=True, readonly=True)]
    ]

    CPU_layout = [
        [sg.Checkbox('Enable Parallel Computing', key='-PARALLEL-', default=True, enable_events=True)],
        [sg.Checkbox('Show the CPU Settings', key='-FLAG-', default=True, enable_events=True)],
        [sg.Column(left_column),sg.Column(right_column)],
        [sg.Button('Submit')]
    ]
    cpu_window =  sg.Window('CPU Configuration for Parallel Computing', CPU_layout, size=(300,300), resizable=True, finalize=True)

    return cpu_window

window = window_layout()

#%%##################--------INITIALIZATION--------########################

#Redirecting Terminal Output to the GUI
output_redirector = OutputRedirector(window, '-OUTPUT-')
sys.stdout = output_redirector
sys.stderr = output_redirector

#Check the second entry of the config file, if it is not empty it shows the path of the Darkframe video
if full_path_dark:
   print(f'Loaded {full_path_dark} as darkframe video')

#initialize the figures for the canvas inside the GUI
px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(250*px,250*px)) #Change the Inset to the canvas her, it depends on the computer you are using it
fig2, ax2 = plt.subplots(1,2,constrained_layout=True, figsize=(10,5))
canvas_elem = window['-CANVAS-']
canvas_elem_pn = window['-PNCANV-']

decoy_img = np.linspace(0, 255, 256*256).reshape(256, 256)

ax.clear()
im = ax.imshow(decoy_img, cmap='gray', vmin=0, vmax=255)
ax.axis('off')
draw_figure(canvas_elem, fig)

i=0
full_path = None
video_data = None
video_pn = None
playing = False
im = None
colorbar = None
batchSize_in = 30

#%%#######################--------MAIN--------###########################

# Event loop
while True:

    # reads the input values of the GUI
    event, values = window.read(timeout=5)
    # reads the slider values for the batch size of the differential view in the video preview field


    # Opens window for CPU Options
    if event == 'CPU Configuration':
        cpu_window = CPU_window_layout()
        event2, value2 = cpu_window.read(timeout=100)

        if event == sg.WINDOW_CLOSED:
            break





    if event == sg.WINDOW_CLOSED:
        break
    
    batchSize = int(values['-BATCH-'])

    # output event for the terminal
    if event == '-OUTPUT-':
        window['-OUTPUT-'].print(values['-OUTPUT-'], end='')

    # reads in the folder and saves the path of the folder in the config file
    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        show_only_npy = values['-SHOW_NPY-']
        file_list = list_files_in_folder(folder, show_only_npy)
        window['-FILELIST-'].update(file_list)
        save_file_paths(folder, full_path_dark)  # Save the selected folder path

    # checkbox event for only showing .npy files
    elif event == '-SHOW_NPY-': # Only show .npy files
        folder = values['-FOLDER-']
        show_only_npy = values['-SHOW_NPY-']
        if folder:
            file_list = list_files_in_folder(folder, show_only_npy)
            window['-FILELIST-'].update(file_list)

    elif event == '-FILELIST-':
        # Update selected file path, no need to change the listbox
        pass
    
    # read in the selected video from the listbox and set all the initial values which are corresponding to the video lenght
    elif event == 'Read Video':
        folder = values['-FOLDER-']
        selected_file = values['-FILELIST-'][0] if values['-FILELIST-'] else None
        if selected_file:
            full_path = os.path.join(folder, selected_file) # Read in the file
            if full_path and os.path.isfile(full_path):
                video_data = np.load(full_path)
                vid_len = video_data.shape[0]
                frame_index = 0
                window['-FRNR-'].update(range=(1, vid_len) ,disabled=False)
                print(f'Loaded video has {vid_len} frames.')
                playing = False  
        else:
            print(f'File could not be loaded: {full_path}')
    
    # option to read in the darkframe video (also from the listbox)
    elif event == 'Select as Darkframe Video':
        folder = values['-FOLDER-']
        selected_file_dark = values['-FILELIST-'][0] if values['-FILELIST-'] else None
        if selected_file_dark:
            full_path_dark = os.path.join(folder, selected_file_dark)
            save_file_paths(folder, full_path_dark)
            dark_frame_video = np.load(full_path_dark)
            print(f'Loaded {full_path_dark} as darkframe video')
        else:
            print(f'File could not be loaded: {full_path_dark}')
    

    elif values['-PLAY OPTION-'] == "Enable Video Player: YES":
        if video_data is not None:
            # checkbox for differential view
            if values['-DIFF-'] == True:
                frame = differential_view(video_data, frame_index, batchSize)
                frame = frame/np.max(frame)
            elif values['-FILTVID-'] == True:
                frame = video_dra[frame_index, :, :]
                frame = frame/np.max(frame)
                vid_len = video_dra.shape[0]
            else:
                frame = video_data[frame_index, :, :]
                frame = frame/np.max(frame)
                vid_len = video_data.shape[0]

            # reset the frame_index advancement
            if event == '-RESET-':
                frame_index = 0
                window['-FRNR-'].update(frame_index)
            # Handle frame navigation
            elif event == 'Next Frame':
                frame_index = (frame_index + 1) % vid_len  # Loop back to start if at end
                window['-FRNR-'].update(frame_index+1)
                playing = False  # Stop automatic play when manually navigating frames
            elif event == 'Previous Frame':
                frame_index = (frame_index - 1) % vid_len  # Loop back to end if at start
                window['-FRNR-'].update(frame_index+1)
                playing = False  # Stop automatic play when manually navigating frames
            elif event == 'Play':
                playing = True
            elif event == 'Stop':
                playing = False

            # slider event for frame advancement
            if event == '-FRNR-' and not playing:
                frame_index = int(values['-FRNR-'])-1

                # Automatically advance frames if playing
            if playing:
                frame_index = (frame_index + 1) % vid_len
                window['-FRNR-'].update(frame_index + 1)
                
            # create figure for the vide preview
            ax.clear()
            im = ax.imshow(frame, cmap='viridis')
            if colorbar is None:
                colorbar = fig.colorbar(im, ax=ax, shrink=0.7)
                colorbar.ax.tick_params(labelsize=5)
            else:
                colorbar.update_normal(im)
            ax.set_title(f'Frame Number {frame_index + 1}')
            ax.axis('off')
            draw_figure(canvas_elem, fig)  # Draw the updated figure

    elif event == '-BATCH_IN-':
        try:
            batchSize_in = int(values['-BATCH_IN-'])
        except:
            print(f'Invalid Batch Size')

    elif event == 'Preprocess Video':
        if video_data is not None:    
            # checkbox for darkframe correction
            if values['-DFC-'] == True:
                mean_dark_frame = np.mean(dark_frame_video)
                video_dfc = np.subtract(video_data, mean_dark_frame)
                print(video_dfc)
                # checkbox for powernormalization of darkframe corrected video
                if values['-PN-'] == True:
                    video_pn, power_fluctuation = Normalization(video_dfc).power_normalized()
                    ax2[0].clear()
                    ax2[0].plot(power_fluctuation, 'b', linewidth=1, markersize=0.3)
                    ax2[0].set_xlabel('Frame #', fontsize=8)
                    ax2[0].set_ylabel(r"$p / \bar p - 1$", fontsize=8)
                    ax2[0].set_title('Intensity fluctuations in the laser beam', fontsize=8)
                    ax2[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    frame_pn = video_pn[1, :, :]
                    ax2[1].clear()
                    ax2[1].set_xlabel(f'Frame Number {frame_index + 1}', fontsize=8)
                    ax2[1].imshow(frame_pn, cmap='gray')
                    ax2[1].set_title('Preprocessed Video', fontsize=8)
                    ax2[1].axis('off')
                    draw_figure(canvas_elem_pn, fig2)

            elif values['-DFC-'] == False:
                # checkbox for powernormalization of pure video without darkframe correction            
                if values['-PN-'] == True:
                    video_pn, power_fluctuation = Normalization(video_data).power_normalized()
                    ax2[0].clear()
                    ax2[0].plot(power_fluctuation, 'b', linewidth=1, markersize=0.3)
                    ax2[0].set_xlabel('Frame #', fontsize=8)
                    ax2[0].set_ylabel(r"$p / \bar p - 1$", fontsize=8)
                    ax2[0].set_title('Intensity fluctuations in the laser beam', fontsize=8)
                    ax2[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    frame_pn = video_pn[1, :, :]
                    ax2[1].clear()
                    ax2[1].set_xlabel(f'Frame Number {frame_index + 1}', fontsize=8)
                    ax2[1].imshow(frame_pn, cmap='gray')
                    ax2[1].set_title('Preprocessed Video', fontsize=8)
                    draw_figure(canvas_elem_pn, fig2)
        else:
            print(f'Please select a video first')


    elif event == 'Find Optimal Batch Size':
        if video_data is not None:
            batch_video = video_data
        elif video_pn is not None:
            batch_video = video_pn
        else:
            print(f'Please select a video first')
        l_range = list(range(10, 50, 10))   
        noise_floor= NoiseFloor(batch_video, list_range=l_range)
        # Optimal value for the batch size
        min_value = min(noise_floor.mean)
        min_index = noise_floor.mean.index(min_value)
        opt_batch = l_range[min_index]
        print(f'Optimal Batch: {opt_batch}')
        window['-BATCH_IN-'].update(opt_batch)
    

    elif event == 'DRA Filtering':
        if video_data is not None:
            print(f'Batch Size is: {batchSize_in}')

            if video_pn is not None:
                video_dra_raw = video_pn
            else:
                video_dra_raw = video_data


            if values['-FPNc-'] == 'DRA_PN':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in)
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
            elif values['-FPNc-'] == 'cpFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='cpFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
            elif values['-FPNc-'] == 'mFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='mFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
            elif values['-FPNc-'] == 'wFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='wFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
            elif values['-FPNc-'] == 'fFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='fFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=False)
        
        else:
            print(f'Please select a video first')

window.close()

# variables for Terminal output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__