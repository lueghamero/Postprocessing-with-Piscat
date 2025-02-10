
import numpy as np
import cv2 as cv
import PySimpleGUI as sg
import os
import sys
import pylab
from piscat.InputOutput import reading_videos
from piscat.Visualization import * 
from piscat.Preproccessing import Normalization
from piscat.BackgroundCorrection import NoiseFloor
from piscat.BackgroundCorrection import DifferentialRollingAverage
from piscat.Localization import *
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
from PSF_localization_preview_copy import *


def window_layout():

    layout = [
        [sg.Text('Piezo Voltage (mV)')],
        [sg.Input(default_text='0', size=(10), key='Volt_IN', enable_events=True)],
        [sg.Button('-1mV'), sg.Button('+1mV')]
    ]

    window = sg.Window('Piezo Voltage Manager', layout, resizable=True, finalize=True)
    
    return window

window = window_layout()

voltage = 0
while True:

    # reads the input values of the GUI
    event, values = window.read(timeout=100)
    
    if event == sg.WINDOW_CLOSED:
        break

    if event == '-1mV':
        voltage -= 1
        window['Volt_IN'].update(voltage)

    elif event == '+1mV':
        voltage += 1
        window['Volt_IN'].update(voltage)
        print(voltage)

window.close()