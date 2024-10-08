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
import time
from mpy_functions_new import *
import pandas

from piscat.Visualization import * 
from piscat.Preproccessing import *
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
from piscat.Trajectory import temporal_filtering
from piscat.Trajectory import particle_linking


#%%##################--------FUNCTIONS--------########################


# creates the figure canvas for the GUI
def draw_figure(canvas_elem, figure, canvas=None):
    
    if canvas is None:
        # If no canvas exists, create a new one
        canvas = FigureCanvasTkAgg(figure, master=canvas_elem.Widget)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    else:
        # If the canvas already exists, just update the drawing
        canvas.draw_idle()
    return canvas

def draw_figure_plot(canvas_elem, figure):

    for widget in canvas_elem.Widget.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(figure, master=canvas_elem.Widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def update_figure(image_data, im, colorbar, psf_positions=None, frame_number=None, radius=8):

    # Clear previous circles if any
    for artist in im.axes.artists:
        artist.remove()  # Remove all previously drawn circles

    im.set_data(image_data)
    
    im.set_clim(vmin=np.min(image_data), vmax=np.max(image_data))
    
    if colorbar:
        colorbar.update_normal(im)

    if psf_positions is not None and frame_number is not None:
        # Filter PSF positions for the current frame
        current_frame_psf = psf_positions[psf_positions[:, 0] == frame_number]
        create_red_circles(current_frame_psf, im.axes, radius=radius)

    im.axes.figure.canvas.draw_idle()

def create_red_circles(psf_positions, ax, radius):

    circles = []
    for psf in psf_positions:
        # PSF positions are assumed to be in the format [frame_num, y, x, sigma]
        y, x = psf[1], psf[2]
        circle = patches.Circle((x, y), radius=radius, edgecolor='red', facecolor='none', lw=1)
        ax.add_patch(circle)
        circles.append(circle)
        
    return circles


#create a folder for the config file, in which saves such as the filepath to the video is stored
CONFIG_DIR = 'config'
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.txt')


# Function to save the folder path to a config file
def save_file_paths(file_path1, file_path2, file_path3, file_path4):
    with open(CONFIG_FILE, 'w') as file:
        file.write(f"{file_path1}\n{file_path2}\n{file_path3}\n{file_path4}")


# Function to load the folder path from a config file
def load_file_paths():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            paths = file.read().strip().split('\n')
            return(
                paths[0] if len(paths) > 0 else '',
                paths[1] if len(paths) > 1 else '',
                paths[2] if len(paths) > 2 else '',
                paths[3] if len(paths) > 3 else '',
            )
    return '', '', '', ''


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


def save_video_in_path(video_data ,input_filepath, output_folder, suffix):

    og_filename = os.path.basename(input_filepath)
    filename_without_ext, ext = os.path.splitext(og_filename)

    new_filename = f'{filename_without_ext}_{suffix}{ext}'

    output_filepath = os.path.join(output_folder, new_filename)

    np.save(output_filepath, video_data)

    print(f'{suffix} video has been saved as {output_filepath}')

def save_dataframe_to_hdf5(df, input_filepath, directory, suffix, key='df'):
    """
    Save a pandas DataFrame to an HDF5 file with a file name based on the directory and suffix.

    Parameters:
    - df: pandas DataFrame to be saved.
    - directory: The directory where the HDF5 file will be saved.
    - suffix: The suffix to append to the file name (before the .h5 extension).
    - key: The key under which the DataFrame is stored in the HDF5 file (default 'df').

    The final file path will be 'directory/suffix.h5'.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create the full file path with suffix
    og_filename = os.path.basename(input_filepath)
    filename_without_ext, _ = os.path.splitext(og_filename)
    file_path = os.path.join(directory, f'{filename_without_ext}_{suffix}.h5')
    
    # Save the DataFrame to the HDF5 file
    df.to_hdf(file_path, key=key, mode='w')
    
    print(f"DataFrame saved to {file_path}")


# Load the last selected folder path
folder, full_path_dark, saving_folder, PSF_folder = load_file_paths()
initial_show_only_npy = True
initial_file_list1 = list_files_in_folder(folder, initial_show_only_npy) if folder else []
dark_frame_video = np.load(full_path_dark) if full_path_dark else []

# Load CPU settings from the config file
cpu_settings = CPUConfigurations()
n_jobs=cpu_settings.n_jobs
backend=cpu_settings.backend
verbose=cpu_settings.verbose,
parallel_active=cpu_settings.parallel_active


#%%###################--------WINDOW_LAYOUT--------######################

#------------------------------------------------------------------------
# Main Window:
def window_layout():

    menu_def = [['&Options', ['&CPU Configuration', '&Inlay Scaling']],['&Data', ['&Save', '&Load', '---','&Info']]]

    vid_prep = [
        [sg.Text("Please select a folder:")],
        [sg.Input(key='-FOLDER-', enable_events=True, default_text=folder), sg.FolderBrowse()],
        [sg.Checkbox('Show only .npy files', key='-SHOW_NPY-', default=initial_show_only_npy, enable_events=True)],
        [sg.Listbox(values=initial_file_list1, size=(60, 10), key='-FILELIST-', auto_size_text=True, enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
        [sg.Button("Read Video"), sg.Button("Crop seleceted Video"), sg.Button("Select as Darkframe Video"),
          sg.OptionMenu(["Enable Video Player: NO", "Enable Video Player: YES"], default_value = "Enable Video Player: NO", key = '-PLAY OPTION-', background_color='lightblue', text_color='black')]
    ]

    vid_play = [
        [sg.Button('Play'), sg.Button('Stop'), sg.Button('Previous Frame'), 
         sg.Slider(range=(1,5000), size=(20,10), orientation='h', key='-FRNR-', enable_events=True, disabled=True), 
         sg.Button('Next Frame'), sg.Button('Reset',key='-RESET-')], 
        [sg.Checkbox("Differential View", key='-DIFF-', enable_events=True), sg.Text('Batch Size:', size=(10,1)), 
         sg.Slider((1, 160), size=(10,10), default_value=1, key='-BATCH-', orientation='horizontal', enable_events=True),
         sg.VerticalSeparator(),
         sg.Checkbox("Show filtered Video", key='-FILTVID-', enable_events=True),
         sg.Checkbox("Show PSFs", key='-SHOW_PSF-', enable_events=True)], 
        [sg.Push(), sg.Canvas(key='-CANVAS-',pad=(0,0)), sg.Push()]
    ]
    
    vid_layout = [
        [sg.Frame("Video Selection", vid_prep, expand_x=True, expand_y=True)],
        [sg.HorizontalLine()],
        [sg.Frame("Video Preview", vid_play, expand_x=True, expand_y=True)]
    ]

    terminal_prep = [  
        [sg.Multiline(size=(150,15), key='-OUTPUT-',expand_x=True, autoscroll = True, background_color='black',
                       text_color='white', reroute_stdout=True, reroute_stderr=False)]
    ]

    power_normalization_df_correction = [
            [sg.Checkbox("Power Normalisation", key='-PN-', enable_events=True)],
            [sg.Checkbox("Darkframe Correction", key='-DFC-', enable_events=True)],
            [sg.Button('Preprocess Video')]

    ]

    DRA = [
        [sg.Column([[sg.Text('FPN correction mode')],
                    [sg.Combo(['DRA_PN', 'cpFPNc', 'mFPNc', 'wFPNc', 'fFPNc'], key='-FPNc-', default_value='DRA_PN', size=(10), enable_events=True, readonly=True)],
                    [sg.Button('DRA Filtering')]],vertical_alignment='top'),
         sg.Column([[sg.Text('Batch Size')], 
                    [sg.Input(default_text='1', size=(10), key='-BATCH_IN-', enable_events=True)], 
                    [sg.Button('Find Optimal Batch Size')]],vertical_alignment='top')]
    ]

    after_dra = [
        [sg.Column([[sg.Checkbox('Temporal Median Filter', key ='-TMPMED-', enable_events = True)],
                    [sg.Checkbox('Flat Field Filter', key ='-FLTFIELD-', enable_events = True)]], vertical_alignment='top'),
         sg.Column([[sg.Button('Apply Filter(s)')]])]
    ]

    piscat_preprocessing = [
        [sg.Column(power_normalization_df_correction, vertical_alignment='top'), sg.VerticalSeparator(), sg.Column(DRA, vertical_alignment='top'), sg.VerticalSeparator(), sg.Column(after_dra, vertical_alignment='top')],
        [sg.Canvas(key='-PNCANV-', pad=(0,0), size=(1000,520))]    
    ]

    psf_detection_input_values_dog = [
        [sg.Frame('DOG/DOH/LOG Values',[
         [sg.Text('Sigma min'), sg.Push(), sg.Slider(range=(1,5), size=(20,10), resolution = 0.5, default_value=1, orientation='h', key='-SMIN-', enable_events=True)],
         [sg.Text('Sigma max'), sg.Push(), sg.Slider(range=(1,10), size=(20,10), resolution = 0.5, default_value=5, orientation='h', key='-SMAX-', enable_events=True)],
         [sg.Text('Sigma ratio'), sg.Push(), sg.Slider(range=(1.1,3), size=(20,10), resolution = 0.1, default_value = 1.1, orientation='h', key='-SSTEP-', enable_events=True)],
         [sg.Text('Threshold'), sg.Push(), sg.Slider(range=(1e-4,1e-2), size=(20,10), resolution= 1e-4, default_value = 5e-3, orientation='h', key='-STHRESH-', enable_events=True)],
         [sg.Text('Overlap'), sg.Push(), sg.Slider(range=(0,1), size=(20,10), resolution= 0.1, default_value=0, orientation='h', key='-OVERLAP-', enable_events=True)],
        ],expand_x=True, expand_y=True)],
        [sg.Frame('DOH Value',
         [[sg.Text('DOH Sigma ratio'), sg.Push(), sg.Slider(range=(1,10), size=(20, 10), resolution=1, default_value=1, orientation='h', key='-SSTEPDOH-', enable_events=True, disabled=True)]]
        ,expand_x=True, expand_y=True)],
        [sg.Frame('RVT Values',[
         [sg.Text('Radius min'), sg.Push(), sg.Slider(range=(1,10), size=(20,10), resolution = 0.5, default_value=1, orientation='h', key='-RMIN-', enable_events=True)],
         [sg.Text('Radius max'), sg.Push(), sg.Slider(range=(1,15), size=(20,10), resolution = 0.5, default_value=5, orientation='h', key='-RMAX-', enable_events=True)],
         [sg.Text('Upsample'), sg.Push(), sg.Slider(range=(1,10), size=(20,10), resolution = 1, default_value = 0, orientation='h', key='-UPSAMPLE-', enable_events=True)],
         [sg.Text('Coarse Factor'), sg.Push(), sg.Slider(range=(1,10), size=(20,10), resolution= 1, default_value = 1, orientation='h', key='-COARSEFACTOR-', enable_events=True)],
        ],expand_x=True, expand_y=True)],
        [sg.Push(), sg.Button('Apply PSF Detection'), sg.Push()]
    ]


    piscat_psf_detection = [
        [sg.Column([ 
         [sg.Text('Detection Mode:')],
         [sg.Combo(['DOG', 'LOG', 'DOH', 'RVT'], key='-PSF_MODE-', default_value='DOG', enable_events=True, size=(20,10),readonly=True)]]),
        sg.Column([[sg.VPush(),sg.Checkbox('PSF Preview', key='-PSFPV-', enable_events=True),sg.VPush()]])],
        [sg.Column(psf_detection_input_values_dog, vertical_alignment='top')],

    ]

    psf_filtering_column1 = [
        [sg.Frame('2D Gauss Fit',[
         [sg.Checkbox('2D-Gaussian Fit', key='-2DGAUSS-', default=True)],
         [sg.Checkbox('FRST Localization Improvement', key='-FRST-', default=True)],
         [sg.Text('ROI Scale'),sg.Push(),sg.Slider(range=(1,20), size=(20,10), resolution=1, default_value=5, orientation='h', key='-GAUSSSCALE-', enable_events=True)],
         [sg.Push(), sg.Button('Apply Gauss Fit'), sg.Push()]],
         expand_x=True, expand_y=True)],
        [sg.Frame('Spatial Filtering', [
         [sg.Checkbox('Outlier Frames', key='-OUTLIER-', default=True)],
         [sg.Text('Threshold Outlier'), sg.Push(), sg.Slider(range=(1,50), size=(20,10), resolution=1, default_value=25, orientation='h', key='-THRESHOUT-', enable_events=True) ],
         [sg.Checkbox('Dense_PSFs', key='-DENSEPSF-', default=True)],
         [sg.Text('Threshold Dense PSFs'), sg.Push(), sg.Slider(range=(0,1), size=(20,10), resolution=0.1, default_value=0, orientation='h', key='-THRESHDENSE-', enable_events=True) ],
         [sg.Checkbox('Symmetric PSFs', key='-SYMMETRICPSF-', default=True)],
         [sg.Text('Threshold Symmetric PSFs'), sg.Push(), sg.Slider(range=(0,1), size=(20,10), resolution=0.1, default_value=0.7, orientation='h', key='-THRESHSYM-', enable_events=True) ],
         [sg.Checkbox('Remove Side Lobes', key='-SIDELOBES-', default=True)],
         [sg.Text('Threshold Side Lobes'), sg.Push(), sg.Slider(range=(0,1), size=(20,10), resolution=0.1, default_value=0, orientation='h', key='-THRESHSIDELOBES-', enable_events=True) ],
         [sg.Push(), sg.Button('Apply Spatial Filters'), sg.Push()]
         ], expand_x=True, expand_y=True)],
    ]

    psf_filtering_column2 = [
        [sg.Frame('Temporal Filtering',[
         [sg.Text('Min Threshold'),sg.Push(),sg.Slider(range=(1,50), size=(20,10), resolution=1, default_value=5, orientation='h', key='-TEMPMIN-', enable_events=True)],
         [sg.Text('Max Threshold'),sg.Push(),sg.Slider(range=(1,300), size=(20,10), resolution=1, default_value=30, orientation='h', key='-TEMPMAX-', enable_events=True)],
         [sg.Push(), sg.Button('Apply Temporal Filters'), sg.Button('Show Histogram'), sg.Push()]],
         expand_x=True, expand_y=True)],

    ]

    psf_filtering = [
        [sg.Column(psf_filtering_column1, vertical_alignment='top'), sg.Column(psf_filtering_column2, vertical_alignment='top', expand_x=True)]
    ]

    tab_layout = [
        [sg.TabGroup([[sg.Tab("Preprocessing", piscat_preprocessing), sg.Tab("PSF Detection", piscat_psf_detection), sg.Tab('PSF Filtering', psf_filtering)]])]
    ]

    piscat_layout =     [
        [sg.Frame("Piscat", tab_layout, expand_x=True, expand_y=True)],
        [sg.HorizontalSeparator()],
        [sg.Frame("History", terminal_prep, expand_x=True)]
    ]

    layout = [
        [sg.Menu(menu_def)],
        [sg.Column(vid_layout, vertical_alignment='top'), sg.VerticalSeparator(), 
         sg.Column(piscat_layout, vertical_alignment='top', expand_x=True, expand_y=True)]
    ]

    # Create the window
    window = sg.Window('iScat Postprocessing using PiScat', layout, size = (1600,900), resizable=True, finalize=True)
    
    return window

#------------------------------------------------------------------------
# Option Window CPU Configuration:

def CPU_window_layout():

    left_column = [
        [sg.Text('Number of Cores:')],
        [sg.Text('Verbosity Level:')],
        [sg.Text('Parallel Method:')]
    ]

    right_column = [
        [sg.Input(default_text=n_jobs, size=(20), key='-N_JOBS-')],
        [sg.Input(default_text=verbose, size=(20), key='-VERBOSE-')],
        [sg.Combo(['loky', 'threading'], key='-BACKEND-', default_value=backend, size=(20), enable_events=True, readonly=True)]
    ]

    CPU_layout = [
        [sg.Checkbox('Enable Parallel Computing', key='-PARALLEL-', default = parallel_active, enable_events=True)],
        [sg.Checkbox('Show the CPU Settings', key='-FLAG-', default=True, enable_events=True)],
        [sg.Column(left_column),sg.Column(right_column)],
        [sg.Button('Submit')],
        [sg.Frame("Description", [[sg.Text('You can choose the number of consecutive workers\n by changing the value for the Number of Cores.\n If the value is -1, all available cores are used.')],
                                  [sg.Text('Verbosity Level gives the amount of output\n in the Terminal while the process is running.\n If it is above 10 everything is shown')],
                                  [sg.Text('Parallel Method is adiviced to be set to loky for\n faster processing and higher stability')],
                                  [sg.Text('Show CPU Settings will produce an output with\n the containment of the JSON config file')],
                                  [sg.Text('ATTENTION!\n If the JSON file is not changed after Submission,\n a restart of the IDE might help')]], expand_x=True, expand_y=True)]

    ]
    cpu_window =  sg.Window('CPU Configuration for Parallel Computing', CPU_layout, resizable=True, finalize=True)

    return cpu_window

#------------------------------------------------------------------------
# Option Window Inlay Scaling:

def Inlay_scaling_layout():

    Inlay_layout = [
        [sg.Text('The size for graphical inlays such as videos or plots might differ, depending on the computer. \n This menu allows you to change the scaling factor \n The standard value 1 is optimized for Mac, on Windows machines 2 should be sufficient')],
        [sg.Push(), sg.Text('Scaling facor:'), sg.Combo(['1', '1.5', '2', '2.5', '3'], key='-SCALING-', default_value='1', size=(10), enable_events = True, readonly=True), sg.Push()],
        [sg.Push(), sg.Button('Submit', key='-SUB_SCALE-'), sg.Push()]
    ]

    scaling_window = sg.Window('Scaling Configuration', Inlay_layout, resizable=True, finalize=True)

    return scaling_window

#------------------------------------------------------------------------
# Data Window Save:

def Saving_layout():
    
    save_layout = [
        [sg.Push(),sg.Text('Choose a Folder for saving Processed Videos'), sg.Push()], 
        [sg.Push(),sg.Input(key='-SAVING_FOLDER-', enable_events=True, default_text=saving_folder), sg.FolderBrowse(),sg.Push()],
        [sg.Checkbox('Save Power Normalized Video', enable_events=True, key='-SAVE_PN-', default=False)],
        [sg.Checkbox('Save DRA Filtered Video', enable_events=True, key='-SAVE_DRA-', default=True)],
        [sg.Checkbox('Delete Video Data from Memory', enable_events=True, key='-DELETE_VID-', default=True)],
        [sg.Push(), sg.Button('Save Video Data'), sg.Button('Save PSF Data'), sg.Push()]
    ]

    saving_window = sg.Window('Save processed Data', save_layout, resizable=True, finalize=True)

    return saving_window


#------------------------------------------------------------------------
# Data Window Load:

def Loading_layout():

    load_layout = [
        [sg.Push(),sg.Text('Choose a video to load'), sg.Push()],
        [sg.Push(),sg.Input(key='-LOADING_FILE-', enable_events=True), sg.FileBrowse(),sg.Push()], 
        [sg.Push(),sg.Button('Load as PN Video'), sg.Button('Load as DRA Video'), sg.Button('Load PSFs Data'),sg.Push()]
    ]

    loading_window = sg.Window('Load processed Data', load_layout, resizable=True, finalize=True)

    return loading_window

#------------------------------------------------------------------------

window = window_layout()

#%%##################--------INITIALIZATION--------########################

#Check the second entry of the config file, if it is not empty it shows the path of the Darkframe video
if full_path_dark:
   print(f'Loaded {full_path_dark} as darkframe video')

#initialize the figures for the canvas inside the GUI
px = 1/plt.rcParams['figure.dpi']  # pixel in inches

fig, ax = plt.subplots()
fig.patch.set_facecolor(color = 'lightblue')
fig2, ax2 = plt.subplots(1,2,constrained_layout=True, figsize=(1000*px,500*px))
canvas_elem = window['-CANVAS-']
canvas_elem_pn = window['-PNCANV-']

decoy_img = np.linspace(0, 255, 256*256).reshape(256, 256)

im = ax.imshow(decoy_img)
ax.axis(False)
canvas = draw_figure(canvas_elem, fig)

i=0
full_path = None
video_data = None
video_pn = None
video_dra = None
playing = False
colorbar = None
im = None
im2 = None
im3 = None
batchSize_in = None
detected_psfs = None
show_psf = None
linked_PSFs_filter = None
loading_file = None


#%%#######################--------MAIN--------###########################

# Event loop (Main window)
while True:

    # reads the input values of the GUI
    event, values = window.read(timeout=100)
    # reads the slider values for the batch size of the differential view in the video preview field

#---------------------------------------------------------
    # Opens window for CPU Options
    if event == 'CPU Configuration':
        cpu_window = CPU_window_layout()
        
        while True:

            event2, value2 = cpu_window.read(timeout=10)

            if event2 == sg.WINDOW_CLOSED:
                break
            
            elif event2 == 'Submit':

                # Enable parallel computing yes or no
                if value2['-PARALLEL-'] == False:
                    parallel_active = False
                elif value2['-PARALLEL-'] == True:
                    parallel_active = True

                # Show the saved settings from the config file
                if value2['-FLAG-'] == True:
                    cpu_set_flag = True
                elif value2['-FLAG-'] == False:
                    cpu_set_flag = True

                # Choose the type of parallel computing (There is also multiprocessing,
                # but it does not work with piscat and I could not solve the problem yet.
                # However loky should be the better choice anyways, as it is more stabel)
                if value2['-BACKEND-'] == 'loky':
                    backend = 'loky'
                elif value2['-BACKEND-'] == 'threading':
                    backend = 'threading'

                n_jobs = int(value2['-N_JOBS-'])
                verbose = int(value2['-VERBOSE-'])

                delete_CPU_config()
                cpu_settings = CPUConfigurations(n_jobs=n_jobs, verbose=verbose, backend=backend, parallel_active=parallel_active, flag_report=cpu_set_flag)
                cpu_window.close()

#---------------------------------------------------------
    # Open Window for adjusting the scale of the canvas 

    if event == 'Inlay Scaling':
        scaling_window = Inlay_scaling_layout()
        
        while True:

            event3, value3 = scaling_window.read(timeout=10)

            if event3 == sg.WINDOW_CLOSED:
                break
            
            elif event3 == '-SUB_SCALE-':

                if value3['-SCALING-'] == '1':
                    pxu = px*1
                elif value3['-SCALING-'] == '1.5':
                    pxu = px*1.5
                elif value3['-SCALING-'] == '2':
                    pxu = px*2
                elif value3['-SCALING-'] == '2.5':
                    pxu = px*2.5
                elif value3['-SCALING-'] == '3':
                    pxu = px*3

    
                fig, ax = plt.subplots(1,1,constrained_layout=True, figsize=(250*pxu,250*pxu)) #Change the Inset to the canvas here, it depends on the computer you are using it on
                fig2, ax2 = plt.subplots(1,2,constrained_layout=True, figsize=(500*pxu,250*pxu))
                ax.clear()
                im = ax.imshow(decoy_img)
                ax.axis('off')
                draw_figure_plot(canvas_elem, fig)

                print(f'Changed Inlay Scaling to: {pxu}')

                scaling_window.close()

#---------------------------------------------------------
    # Open Window for Saving Data
    if event == 'Save':
        saving_window = Saving_layout()

        while True:
            
            event4, value4 = saving_window.read(timeout=10)

            if event4 == sg.WINDOW_CLOSED:
                break
                
            elif event4  == '-SAVING_FOLDER-':
                saving_folder = value4['-SAVING_FOLDER-']
                save_file_paths(folder, full_path_dark, saving_folder, PSF_folder) 

            elif event4 == 'Save Video Data':

                if value4['-SAVE_PN-'] == True:
                    if video_pn is not None:
                        if value4['-DELETE_VID-'] == False:
                            save_video_in_path(video_pn, full_path, saving_folder, suffix='power_normalized')
                            video_pn = None
                        else:
                            save_video_in_path(video_pn, full_path, saving_folder, suffix='power_normalized')
                    else:
                        print(f'There is no Power Normalized Video in the Memory')


                if value4['-SAVE_DRA-'] == True:
                    if video_dra is not None:
                        if value4['-DELETE_VID-'] == False:
                            save_video_in_path(video_dra, full_path, saving_folder, suffix='DRA_filtered')
                            video_dra = None
                        else:
                            save_video_in_path(video_dra, full_path, saving_folder, suffix='DRA_filtered')
                    else:
                        print(f'There is no DRA Filtered Video in the Memory')
                    pass
            
            elif event4 == "Save PSF Data":
                if full_path is not None:
                    input_path = full_path
                else: 
                    input_path = loading_file
                
                if detected_psfs is not None:
                    save_dataframe_to_hdf5(detected_psfs, input_path, saving_folder, suffix = 'detected_PSFs')

                elif linked_PSFs_filter is not None:
                    save_dataframe_to_hdf5(linked_PSFs_filter, input_path, saving_folder, suffix = 'detected_PSFs')
                
                else: 
                    print(f'No PSFs detected yet. Pls load Data or process the Video')

#---------------------------------------------------------
    # Open Window for Loading Data
    if event == 'Load':
        loading_window = Loading_layout()

        while True:

            event5, value5 = loading_window.read(timeout=10)

            if event5 == sg.WINDOW_CLOSED:
                break

            elif event5  == '-LOADING_FILE-':
                loading_file = value5['-LOADING_FILE-']

            elif event5 == 'Load as PN Video':
                video_pn = np.load(value5['-LOADING_FILE-'])
                print(f'Loaded {loading_file} as PN video')
                loading_window.close()

            elif event5 == 'Load as DRA Video':
                video_dra = np.load(value5['-LOADING_FILE-'])
                print(f'Loaded {loading_file} as DRA video')
                PSFs = PSFsExtraction(video_dra)
                vid_len = video_dra.shape[0]
                frame_index = 0
                window['-FRNR-'].update(range=(1, vid_len) ,disabled=False)
                loading_window.close()

            elif event5 == 'Load PSFs Data':
                detected_psfs = pandas.read_hdf(value5['-LOADING_FILE-'], key='df')
                print(f'Loaded {loading_file} as PSFs Data Frame')
                loading_window.close()

#---------------------------------------------------------
# Main window again

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
        save_file_paths(folder, full_path_dark, saving_folder, PSF_folder)  # Save the selected folder path

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
            save_file_paths(folder, full_path_dark, saving_folder, PSF_folder)
            dark_frame_video = np.load(full_path_dark)
            print(f'Loaded {full_path_dark} as darkframe video')
        else:
            print(f'File could not be loaded: {full_path_dark}')

    elif event == 'Crop seleceted Video':
        if 'full_path' in locals() and full_path:    
            cropper = vc(full_path)
            cropper.select_roi(frame_index=10)
            cropped_video_data = cropper.crop_video()
            cropper.save_cropped_video(cropped_video_data)
        else:
            sg.popup('Please load a video file first.')
    

    elif values['-PLAY OPTION-'] == "Enable Video Player: YES":
        if video_data is not None or video_dra is not None:
            
            if video_data is None:
                video_data = video_dra

            # checkbox for differential view
            if values['-DIFF-'] == True:
                frame = differential_view(video_data, frame_index, batchSize)
                frame = frame/np.max(frame)
            elif values['-FILTVID-'] == True:
                if video_dra is None:
                    print(f'Filtered video is not yet in memory! Please process first!')
                    window['-FILTVID-'].update(False) 
                    continue
                else:
                    frame = video_dra[frame_index, :, :]
                    frame = frame/np.max(frame)
                    vid_len = video_dra.shape[0]
                    window['-FRNR-'].update(range=(1, vid_len) ,disabled=False)
                    if values['-SHOW_PSF-'] == True:
                        if detected_psfs is not None:
                            show_psf = detected_psfs[['frame','y','x','sigma','center_intensity']].to_numpy()
                            if linked_PSFs_filter is not None:
                                show_psf = linked_PSFs_filter[['frame','y','x','sigma','center_intensity']].to_numpy()
                        else:
                            print(f'No PSFs detected yet. Pls load Data or process the Video')
                            window['-SHOW_PSF-'].update(False)
                            continue 
            else:
                frame = video_data[frame_index, :, :]
                frame = frame/np.max(frame)
                vid_len = video_data.shape[0]
                window['-FRNR-'].update(range=(1, vid_len) ,disabled=False)


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


            [p.remove() for p in reversed(ax.patches)]

            if im is None:
                # If it's the first frame, create the image and colorbar
                im = ax.imshow(frame, cmap='viridis')
                ax.set_title(f'Frame Number {frame_index + 1}')
                ax.axis('off')
                colorbar = fig.colorbar(im, ax=ax, shrink=0.7)
                colorbar.ax.tick_params(labelsize=5)
                fig.tight_layout()
            else:
                if values['-SHOW_PSF-'] == True:
                    # Update the image data on subsequent frames
                    ax.set_title(f'Frame Number {frame_index + 1}')
                    update_figure(frame, im, colorbar, psf_positions = show_psf, frame_number = frame_index, radius=8)
                else:
                    ax.set_title(f'Frame Number {frame_index + 1}')
                    update_figure(frame, im, colorbar)    
            # Draw or update the figure on the canvas
            canvas = draw_figure(window['-CANVAS-'], fig, canvas)  

        else:
            print(f'No video for playing in memory')
            window['-PLAY OPTION-'].Update("Enable Video Player: NO")
            continue

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
                    draw_figure_plot(canvas_elem_pn, fig2)

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
                    draw_figure_plot(canvas_elem_pn, fig2)
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
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
            elif values['-FPNc-'] == 'cpFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='cpFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
            elif values['-FPNc-'] == 'mFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='mFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
            elif values['-FPNc-'] == 'wFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='wFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)
            elif values['-FPNc-'] == 'fFPNc':
                video_dr = DifferentialRollingAverage(video_dra_raw, batchSize_in, mode_FPN='fFPN')
                video_dra, _ = video_dr.differential_rolling(FPN_flag=True, select_correction_axis='Both', FFT_flag=True)

            PSFs = PSFsExtraction(video_dra)

        else:
            print(f'Please select a video first')

    elif event == 'Apply Filter(s)':
        if video_dra is not None:
            pre_filters = Filters(video = video_dra)
            if values['-TMPMED-'] == True:
                video_dra = pre_filters.temporal_median()
            if values['-FLTFIELD-'] == True:
                video_dra = pre_filters.flat_field(sigma = 1.1)
        else:
            print(f'DRA filtered video is not yet in memory! Please process or load first!')
            window['-PSFPV-'].update(False)
            continue


    elif values['-PSFPV-'] == True:
        window['-PLAY OPTION-'].update("Enable Video Player: NO")

        if video_dra is not None:
            
            if values['-PSF_MODE-'] == 'DOG':
                function = 'dog'
                window['-SSTEPDOH-'].update(disabled=True)
            elif values['-PSF_MODE-'] == 'DOH':
                function = 'doh'
                window['-SSTEPDOH-'].update(disabled=False)
            elif values['-PSF_MODE-'] == 'LOG':
                function = 'log'
                window['-SSTEPDOH-'].update(disabled=True)                
            elif values['-PSF_MODE-'] == 'RVT':
                function = 'RVT'
                window['-SSTEPDOH-'].update(disabled=True)

            if values['-PSF_MODE-'] == 'DOH':
                sigma_ratio = int(values['-SSTEPDOH-'])
            else:
                sigma_ratio = float(values['-SSTEP-'])

            min_sigma = float(values['-SMIN-'])
            max_sigma = float(values['-SMAX-'])
            threshold = float(values['-STHRESH-'])
            overlap = float(values['-OVERLAP-'])

            frame_index = int(values['-FRNR-'])-1
            frame = video_dra[frame_index,:,:]

            psf_positions = PSFs.psf_detection_preview(frame_number = frame_index, function=function,
                                                       min_sigma = min_sigma, max_sigma = max_sigma,
                                                       sigma_ratio = sigma_ratio, threshold = threshold, overlap = overlap)

            psf_preview = psf_positions[['frame','y','x','sigma','center_intensity']].to_numpy()
            
            [p.remove() for p in reversed(ax.patches)]  

            circles = []
            if im is None:
                # If it's the first frame, create the image and colorbar
                im = ax.imshow(frame, cmap='viridis')
                ax.set_title(f'Frame Number {frame_index + 1}')
                ax.axis('off')
                colorbar = fig.colorbar(im, ax=ax, shrink=0.7)
                colorbar.ax.tick_params(labelsize=5)
                fig.tight_layout()

            else:
                # Update the image data on subsequent frames
                ax.set_title(f'Frame Number {frame_index + 1}')
                update_figure(frame, im, colorbar, psf_positions=psf_preview, frame_number = frame_index, radius=8)

            canvas = draw_figure(window['-CANVAS-'], fig, canvas)
            
        else:
            print(f'Filtered video is not yet in memory! Please process or load first!')
            window['-PSFPV-'].update(False)
            continue
    
    elif event == 'Apply PSF Detection':
        
        if video_dra is not None:
            
            if values['-PSF_MODE-'] == 'DOG':
                function = 'dog'
            elif values['-PSF_MODE-'] == 'DOH':
                function = 'doh'
            elif values['-PSF_MODE-'] == 'LOG':
                function = 'log'               
            elif values['-PSF_MODE-'] == 'RVT':
                function = 'RVT'

            if values['-PSF_MODE-'] == 'DOH':
                sigma_ratio = int(values['-SSTEPDOH-'])
            else:
                sigma_ratio = float(values['-SSTEP-'])

            min_sigma = float(values['-SMIN-'])
            max_sigma = float(values['-SMAX-'])
            threshold = float(values['-STHRESH-'])
            overlap = float(values['-OVERLAP-'])

            rmin = values['-RMIN-']
            rmax = values['-RMAX-']
            upsample = values['-UPSAMPLE-']
            coarse_factor = values['-COARSEFACTOR-']

            detected_psfs = PSFs.psf_detection(function = function, min_sigma = min_sigma, max_sigma = max_sigma,
                                               sigma_ratio = sigma_ratio, threshold = threshold, overlap = overlap,
                                               min_radial = rmin, max_radial = rmax, upsample = upsample,
                                               coarse_factor = coarse_factor)
            
            print(f'Done!')

        else:
            print(f'Filtered video is not yet in memory! Please process or load first!')
            window['-PSFPV-'].update(False)
            continue
    
    elif event == 'Apply Gauss Fit':
        if detected_psfs is not None:

            gaussscale = int(values['-GAUSSSCALE-'])

            if values['-2DGAUSS-']== True:
                detected_psfs = PSFs.fit_Gaussian2D_wrapper(detected_psfs, scale = gaussscale, internal_parallel_flag = parallel_active)
            if values['-FRST-'] == True:
                detected_psfs = PSFs.improve_localization_with_frst(detected_psfs, scale = gaussscale)

        else:
            print(f'No PSFs detected yet. Pls load Data or process the Video')

    elif event == 'Apply Spatial Filters':
        if detected_psfs is not None:    
            
            filters = SpatialFilter()

            threshout = int(values['-THRESHOUT-'])
            threshdense = float(values['-THRESHDENSE-'])
            threshsym = float(values['-THRESHSYM-'])
            threshsidelobes = float(values['-THRESHSIDELOBES-'])

            if values['-OUTLIER-'] == True:
                detected_psfs = filters.outlier_frames(detected_psfs, threshold = threshout)
            if values['-DENSEPSF-'] == True:
                detected_psfs = filters.dense_PSFs(detected_psfs, threshold = threshdense)
            if values['-SYMMETRICPSF-'] == True:
                detected_psfs = filters.symmetric_PSFs(detected_psfs, threshold = threshsym)
            if values['-SIDELOBES-'] == True:
                detected_psfs = filters.remove_side_lobes_artifact(detected_psfs, threshold = threshsidelobes)

        else:
            print(f'No PSFs detected yet. Pls load Data or process the Video')

    elif event == 'Apply Temporal Filters':
        if detected_psfs is not None:

            linking = particle_linking.Linking()
            linked_psfs = linking.create_link(psf_position=detected_psfs, search_range = 2, memory = 10)

            t_filters = temporal_filtering.TemporalFilter(video = video_dra, batchSize = batchSize_in)

            temp_thresh_min = int(values['-TEMPMIN-'])
            temp_thresh_max = int(values['-TEMPMAX-'])

            all_trajectories, linked_PSFs_filter, his_all_particles = t_filters.v_trajectory(df_PSFs = linked_psfs,
                                                                                              threshold_min = temp_thresh_min,
                                                                                              threshold_max = temp_thresh_max)

            print(all_trajectories)
            print(linked_PSFs_filter)
        else:
            print(f'No PSFs detected yet. Pls load Data or process the Video')
    
    elif event == 'Show Histogram':
        if his_all_particles is not None:
            his_all_particles = linked_psfs['particle'].value_counts()

            fig3 = plt.figure(figsize=(12, 4))

            plt.hist(his_all_particles, bins=50, fc='C1', ec='k', density=False)
            plt.ylabel('#Counts', fontsize=18)
            plt.xlabel('Length of linking', fontsize=18)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.title('Linking', fontsize=18)
            print('Median of linking length is {}'.format(np.median(his_all_particles)))
            plt.figure(fig3.number)
            plt.show()

        else:
            print(f'Temporal filtering was not conducted yet')


window.close()
