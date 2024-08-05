import os
import PySimpleGUI as sg

CONFIG_DIR = 'config'
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.txt')

# Ensure the config directory exists
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

# Function to list files in a folder
def list_files_in_folder(folder):
    try:
        return os.listdir(folder)
    except FileNotFoundError:
        return ["Folder not found."]
    except NotADirectoryError:
        return ["Selected path is not a folder."]
    except PermissionError:
        return ["Permission denied."]

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

# Load the last selected folder path
initial_folder = load_folder_path()
initial_file_list = list_files_in_folder(initial_folder) if initial_folder else []

# Define the window's contents
layout = [
    [sg.Text("Please select a folder:")],
    [sg.Input(key='-FOLDER-', enable_events=True, default_text=initial_folder), sg.FolderBrowse()],
    [sg.Listbox(values=initial_file_list, size=(60, 15), key='-FILELIST-', enable_events=True, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
    [sg.Button("Read File"), sg.Button("Cancel")]
]

# Create the window
window = sg.Window('Folder and File Selector', layout)

# Event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    if event == "Cancel":
        break
    if event == '-FOLDER-':
        folder = values['-FOLDER-']
        file_list = list_files_in_folder(folder)
        window['-FILELIST-'].update(file_list)
        save_folder_path(folder)  # Save the selected folder path
    elif event == '-FILELIST-':
        selected_file = values['-FILELIST-'][0]
        # No need to update the listbox here, just store the selected file
    elif event == 'Read File':
        folder = values['-FOLDER-']
        selected_file = values['-FILELIST-'][0] if values['-FILELIST-'] else None
        if selected_file:
            full_path = os.path.join(folder, selected_file)
            if os.path.isfile(full_path):
                with open(full_path, 'r') as file:
                    file_content = file.read()
                sg.popup_scrolled(f'Contents of {selected_file}:\n\n{file_content}')
            else:
                sg.popup(f'File not found: {full_path}')
    

# Close the window
window.close()



