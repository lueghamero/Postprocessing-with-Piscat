import PySimpleGUI as sg
import sys
import time
import threading
from io import StringIO

# Redirect stdout and stderr
class OutputRedirector:
    def __init__(self, window, key):
        self.window = window
        self.key = key

    def write(self, message):
        self.window.write_event_value(self.key, message)

    def flush(self):
        pass

# Function to simulate code execution and print output
def long_running_task():
    for i in range(10):
        print(f"Task is running: Step {i+1}")
        time.sleep(1)
    print("Task completed.")

# PySimpleGUI layout
layout = [
    [sg.Multiline(size=(80, 20), key='-OUTPUT-', autoscroll=True, reroute_stdout=True, reroute_stderr=True)],
    [sg.Button('Start Task')]
]

# Create the window
window = sg.Window('Terminal Output', layout, finalize=True)

# Redirect stdout and stderr
output_redirector = OutputRedirector(window, '-OUTPUT-')
sys.stdout = output_redirector
sys.stderr = output_redirector

# Event loop
while True:
    event, values = window.read(timeout=100)
    
    if event == sg.WINDOW_CLOSED:
        break

    if event == 'Start Task':
        threading.Thread(target=long_running_task, daemon=True).start()

    if event == '-OUTPUT-':
        window['-OUTPUT-'].print(values['-OUTPUT-'])

# Close the window
window.close()

# Reset stdout and stderr to their original values
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


