# tutorial 3 
import numpy as np
from piscat.Visualization import * 
from piscat.Preproccessing import * 
from piscat.BackgroundCorrection import *
from piscat.InputOutput import *
from piscat.Localization import *
import matplotlib.pyplot as plt


demo_video_path = r"C:\Users\Dante\Desktop\DNA-PAINT\Control\control_4999_128_128_uint16_2.33FPS.raw"


video = video_reader(file_name=demo_video_path, type='binary', img_width=128, img_height=128,
                                    image_type=np.dtype('<u2'), s_frame=0, e_frame=-1)#Loading the video

status_ = read_status_line.StatusLine(video)#Reading the status line
video_remove_status, status_information  = status_.find_status_line()#Examining the status line & removing it

DRA_PN_cpFPNc = DifferentialRollingAverage(video=video_remove_status, batchSize=120, mode_FPN='cpFPN')
RVideo_PN_cpFPNc, gainMap1D_cpFPN = DRA_PN_cpFPNc.differential_rolling(FPN_flag=True,
                                                                      select_correction_axis='Both',
                                                                      FFT_flag=False)

Display(RVideo_PN_cpFPNc, time_delay=10)