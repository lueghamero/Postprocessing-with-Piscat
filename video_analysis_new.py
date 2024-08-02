# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:04:54 2023

@author: Clara from github 24.04.23
"""

#%% 
from mpy_functions_new import *
from scipy import ndimage
from skimage import io
from matplotlib.widgets import Button

import PySimpleGUI as sg
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import pandas as pd 
import re

version = 'v3'


px = 1/plt.rcParams['figure.dpi']  # pixel in inches

plt.rcParams.update({'font.size': 800*px})
sg.theme('Dark Teal 10')

#scaling = get_scaling()
scaling = 2


space_size = (int(5/scaling),1)
text_size = (int(20/scaling),1)
slider_size = (int(20/scaling),5)

# --- PySimpleGUI Windows --- #

def make_window_view(max_length):

    col1 = [
            
        [ sg.Text(' ', size = space_size), sg.Text('Frames to Average', size = text_size), sg.Slider((1, 1000),       orientation = 'h', s = slider_size, key = '-avg-', default_value = (1) ) ],
        [ sg.Text(' ', size = space_size), sg.Text('Current Frame',     size = text_size), sg.Slider((1, max_length), orientation = 'h', s = slider_size, key = '-pos-', default_value = (1), enable_events = True ) ]
        
        ]

    col2 = [
        
        [ sg.Text(' ', size = space_size), sg.Text('FT Filter Low',       size = text_size), sg.Slider((1, 20),  orientation = 'h', s = slider_size, key = '-ft_low-',  default_value = (1)   ) ], 
        [ sg.Text(' ', size = space_size), sg.Text('FT Filter High',      size = text_size), sg.Slider((1, 200), orientation = 'h', s = slider_size, key = '-ft_high-', default_value = (100) ) ],
        [ sg.Text(' ', size = space_size), sg.Text('FT Filter Max', size = text_size), sg.Slider((1, 500), orientation = 'h', s = slider_size, key = '-ft_th-',   default_value = (500) ) ]
            
        ]

    col3 = [
            
        [ sg.Text(' ', size = space_size), sg.OptionMenu(values = ('Differential', 'Mean Filter', 'Video'),                                          key = '-view_opt-' , default_value = ('Video') ) ], 
        [ sg.Text(' ', size = space_size), sg.OptionMenu(values = ('Noise Map: Rows', 'Noise Map: Columns', 'Noise Map: Both', 'Noise Map: None'),   key = '-n_map-'    , default_value = ('Noise Map: Rows') )], 
        [ sg.Text(' ', size = space_size), sg.OptionMenu(values = ('Blob Detection Off', 'Blob Detection On'),                                       key = '-blob-'     , default_value = ('Blob Detection Off') ) ]
        
        ]

    col4 = [

        [ sg.Text(' ', size = space_size), sg.OptionMenu(values = ('Tracking On', 'Tracking Off'), key = '-track_opt-' , default_value = ('Tracking Off') ) ], 
        [ sg.Text(' ', size = space_size), sg.Text('PD Threshold', size = text_size), sg.Slider( (0.005, 0.1), orientation = 'h', s = slider_size, key = '-p_th-', default_value = (0.03), resolution=.005 ) ], 
        [ sg.Text(' ', size = space_size), sg.Text('Color Bar',    size = text_size), sg.Slider( (0.5, 30),     orientation = 'h', s = slider_size, key = '-cb-',   default_value = (5), resolution=.5   ) ]

    ]

    col5 = [
        [sg.Text(' ', size = space_size), sg.Checkbox('Particle Detection',  key = '-find_p-',   default = False)], 
        [sg.Text(' ', size = space_size), sg.Checkbox('Save Video',          key = '-save_vid-', default = True)],
        [sg.Text(' ', size = space_size)], 
        [sg.Text(' ', size = space_size), sg.Button('Continue')]

    ]

    row2 = [ 
        
        [sg.Canvas(key='-CANVAS-')]
        
        ]

    layout = [

        [ sg.Column(col1), sg.Column(col2), sg.Column(col3), sg.Column(col4), sg.Column(col5) ],      
        [ sg.Column(row2, justification='center') ]
    
    ]

    window = sg.Window("View Differential", layout, finalize=True, size = (1400,800), resizable=True)
    window['-avg-'].bind('<ButtonRelease-1>', ' Release')

    return window

####################################################################
########################### --- Main --- ###########################       
####################################################################

#%%

# --- Get the file --- #
filename = sg.popup_get_file('Filename to play')
#filename = r"C:\Users\Emanuel\Desktop\Masterarbeit\2024_02_27_data\12_59_32_Sample1_Refrence_Air.npy"

if filename is None:
    exit()
#My version of extracting folder in which data is stored and name of data:
filename_folder = os.path.dirname(filename)
filename_measurement = os.path.splitext(os.path.basename(filename))[0]
#filename_folder = filename[0:re.search("data", filename).start()+17]
#filename_measurement = filename[re.search("data", filename).start()+17:-10]

vidFile    = load_npy(filename)
vidFile = np.transpose(vidFile, (1, 2, 0))
vid        = vid_norm(vidFile)
length_vid = np.shape(vid)[2]

#%%

# --- View Video --- #
window_view        = make_window_view(length_vid)
event, view_values = window_view.read(timeout = 10)

avg      = 1
view_opt = 'Differential'
nmap_opt = 'Noise Map: None'
blob_opt = 'Blob Dection Off'
cb_min = -0.1/np.sqrt(avg)
cb_max = 0.1/np.sqrt(avg)
f_cb = 1
i        = 0

disp = view_ini(vid, avg, i, view_opt, nmap_opt)
disp_ft, disp_filt_ft, disp_filt = ft_filter(disp/avg, 1, 100, 100)

fig, axarr = plt.subplots(1,3, constrained_layout = True, figsize=(2000*px,2000*px)) 
fig_agg = draw_figure(window_view['-CANVAS-'].TKCanvas, fig)
fig.patch.set_facecolor(color = 'paleturquoise')

axarr[0].set_title('Video')
im = axarr[0].imshow(disp/avg, vmin = cb_min, vmax = cb_max)
fig.colorbar(im, ax = axarr[0], shrink=0.48, aspect = 50, format = ticker.FormatStrFormatter('% 2.4f'))

axarr[1].set_title('FT')
ft = axarr[1].imshow(abs(disp_filt_ft), vmin = 0, vmax = np.max(np.abs(disp_filt_ft)))
fig.colorbar(ft, ax = axarr[1], shrink=0.48, aspect = 50, format = ticker.FormatStrFormatter('% 2.2f'))

axarr[2].set_title('FT filtered Video')  
filt = axarr[2].imshow(disp_filt, vmin = cb_min, vmax = cb_max)
fig.colorbar(filt, ax = axarr[2], shrink=0.48, aspect = 50, format = ticker.FormatStrFormatter('% 2.4f'))

while True:
    i = i + 1

    event, view_values = window_view.read(timeout = 10)
    blob_opt = view_values['-blob-']

    if event == sg.WINDOW_CLOSED:
        quit()        

    elif event == 'Continue':
        break

    elif i-1+2*avg >= length_vid:
        i = 1
        disp = view_ini(vid, avg, i, view_opt, nmap_opt)

    elif event == '-avg- Release' or i-1 != view_values['-pos-'] or view_opt != view_values['-view_opt-'] or nmap_opt != view_values['-n_map-'] or f_cb != view_values['-cb-']:
        avg = int(view_values['-avg-'])
        i = int(view_values['-pos-'])
        view_opt = view_values['-view_opt-']
        nmap_opt = view_values['-n_map-']
        f_cb = view_values['-cb-']

        disp = view_ini(vid, avg, i, view_opt, nmap_opt)

        if view_opt == 'Differential': 

            disp_ft, disp_filt_ft, disp_filt = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))
            
            cb_min = -0.1/np.sqrt(avg)*f_cb
            cb_max = 0.1/np.sqrt(avg)*f_cb
            
            im.set_data(disp/avg)
            ft.set_data(abs(disp_filt_ft))
            filt.set_data(disp_filt)

        else: # Mean or normal

            disp_ft, disp_filt_ft, disp_filt = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

            cb_min = np.mean(np.absolute(disp))*-f_cb
            cb_max = np.mean(np.absolute(disp))*f_cb

            im.set_data(disp)
            ft.set_data(abs(disp_filt_ft))
            filt.set_data(disp_filt)
        
        im.set_clim(vmin = cb_min, vmax = cb_max)
        ft.set_clim(vmin = 0, vmax = np.max(np.abs(disp_filt_ft))) 
        filt.set_clim(vmin = cb_min, vmax = cb_max) 

    else:

        disp = view_cont(disp, vid, avg, i, view_opt, nmap_opt)

        if view_opt == 'Differential':

            disp_ft, disp_filt_ft, disp_filt = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

            im.set_data(disp/avg)
            ft.set_data(abs(disp_filt_ft))
            filt.set_data(disp_filt)

        else: # Mean or normal 
            
            disp_ft, disp_filt_ft, disp_filt = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

            im.set_data(disp)
            ft.set_data(abs(disp_filt_ft))
            filt.set_data(disp_filt)

        if blob_opt == 'Blob Detection On':
            if view_opt == 'Differential':
                p_th = view_values['-p_th-']/np.sqrt(avg)
            else: 
                p_th = view_values['-p_th-']
            particles = find_blobs(disp_filt, p_th, i)

            for row in particles: 
                cir = patches.Circle((row[2], row[1]), 8, color='r',fill = False)
                axarr[2].add_patch(cir)

    window_view.Element('-pos-').Update(i)  
    fig_agg.draw()   
    [p.remove() for p in reversed(axarr[2].patches)]    
    
plt.close()
window_view.close()

#%%

if view_values['-save_vid-'] == True: 

    name_str = '___avg' + str(avg) + '_' + view_opt + '_nmap' + nmap_opt[11:] + '_Low' + str(int(view_values['-ft_low-'])) + '_High' + str(int(view_values['-ft_high-'])) + '_Th' + str(int(view_values['-ft_th-']))
    print('Saving ' + name_str)

    path = os.path.join(filename_folder, 'mp_scat/videos')
    make_folder(path)
    fn = path + '/' + version + '_' + filename_measurement + name_str + '.mp4'

    disp = view_ini(vid, avg, 0, view_opt, nmap_opt)

    if view_opt == 'Differential':
        disp_ft, disp_filt_ft, img = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

    else: # Mean or normal 
        disp_ft, disp_filt_ft, img = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))


    fig,ax = plt.subplots()
    im_cb = ax.imshow(disp, cmap='gray')
    im_cb.set_clim(vmin = cb_min, vmax = cb_max)
    cbar = fig.colorbar(im_cb, ax = ax)
    cbar.ax.tick_params(labelsize = 20) 
    ax.remove()
    plt.savefig('temp_colorbar.png',bbox_inches='tight')

    cb_img = io.imread('temp_colorbar.png', as_gray=True)*255
    h_cb = int(np.shape(vid)[0])
    w_cb = int(np.round(cb_img.shape[1]*(h_cb/cb_img.shape[0])))
    
    cb_img = cv2.resize(cb_img, (w_cb, h_cb), interpolation = cv2.INTER_AREA)

    size = np.shape(vid)[0], np.shape(vid)[1] + w_cb
    fps = 1000
    out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    
    for i in range(1,length_vid-2*avg,1):
        progressbar(i, length_vid-2*avg, suffix = 'saving')

        disp = view_cont(disp, vid, avg, i, view_opt, nmap_opt)

        if view_opt == 'Differential':
            disp_ft, disp_filt_ft, img = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

        else: # Mean or normal 
            disp_ft, disp_filt_ft, img = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))


        img = ((img+abs(cb_min))/(cb_max+abs(cb_min)))*255
        img[img > 255] = 255
        img[img < 1] = 1
        img_conc = np.concatenate((img, cb_img), axis=1)

        data = img_conc.astype(np.uint8)
        out.write(data)

    cv2.destroyAllWindows()
    out.release()

#%%

if view_values['-find_p-'] == True:

    disp = view_ini(vid, avg, 0, view_opt, nmap_opt)

    if view_opt == 'Differential':
        disp_ft, disp_filt_ft, img = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

    else: # Mean or normal 
        disp_ft, disp_filt_ft, img = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

    img = ndimage.median_filter(img,3)
    particles = find_blobs(img, p_th, 0)

    for i in range(1,length_vid-2*avg,1):
        progressbar(i, length_vid-2*avg, suffix = 'particle detection')

        disp = view_cont(disp, vid, avg, i, view_opt, nmap_opt)

        if view_opt == 'Differential':
            disp_ft, disp_filt_ft, img = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

        else: # Mean or normal 
            disp_ft, disp_filt_ft, img = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))
        
        img = ndimage.median_filter(img,3)
        new_p = find_blobs(img, p_th, i)

        particles = np.append(particles, new_p, axis = 0)
        
    print(particles)

    name_str = '___avg' + str(avg) + '_' + view_opt + '_nmap' + nmap_opt[11:] + '_Low' + str(int(view_values['-ft_low-'])) + '_High' + str(int(view_values['-ft_high-'])) + '_ftTh' + str(int(view_values['-ft_th-'])) + '_pTh' + str((np.round(p_th,4))).replace('.', '_')

    path = os.path.join(filename_folder, 'mp_scat/particles')
    make_folder(path)
    fn = version + '_' + filename_measurement + name_str

    np.save((os.path.join(path, fn)+'_particles'), particles)


    all_p = group_p(particles)

    p_contrasts = np.empty(np.shape(all_p)[0], object)
    c_max = np.empty(np.shape(all_p)[0]) 
    c_gaus = np.empty(np.shape(all_p)[0])
    c_ind = np.empty(np.shape(all_p)[0])
    c_pos = np.empty((2, np.shape(all_p)[0]))
    x_posish = np.empty(np.shape(all_p)[0])
    y_posish = np.empty(np.shape(all_p)[0])

    n = 0
    #%%

    for row in all_p: 

        p_contrasts[n] = particles[ ( abs(particles[:, 1] - row[0]) < 6 ) & ( abs(particles[:, 2] - row[1]) < 6 ), :] 
        c_max[n] = np.amax(np.abs(p_contrasts[n][:,4]))
        max_i = np.argmax(np.abs(p_contrasts[n][:,4]))

        c_ind[n] = int(p_contrasts[n][max_i,0])
        c_pos[0,n] = int(p_contrasts[n][max_i,2])
        c_pos[1,n] = int(p_contrasts[n][max_i,1])

        disp = view_ini(vid, avg, int(c_ind[n]), view_opt, nmap_opt)

        if view_opt == 'Differential':
            disp_ft, disp_filt_ft, img = ft_filter(disp/avg, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

        else: # Mean or normal 
            disp_ft, disp_filt_ft, img = ft_filter(disp, int(view_values['-ft_low-']), int(view_values['-ft_high-']), int(view_values['-ft_th-']))

        def analyse(event): 
            plt.close()

        def skip(event): 
            global skip_fit
            skip_fit = True
            plt.close()

        skip_fit = False

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        plt.imshow(img)
        cir = patches.Circle((c_pos[0,n], c_pos[1,n]), 8, color='r',fill = False)
        ax.add_patch(cir)

        axskip = plt.axes([0.7, 0.05, 0.1, 0.075])
        axanalyse = plt.axes([0.81, 0.05, 0.1, 0.075])
        banalyse = Button(axanalyse, 'Analyse')
        banalyse.on_clicked(analyse)
        bskip = Button(axskip, 'Skip')
        bskip.on_clicked(skip)

        plt.show()

        if skip_fit == True: 
            print('skip')
            
            c_max[n]  = 0
            c_gaus[n] = 0
            c_ind[n]  = 0
            c_pos[0,n] = 0
            c_pos[1,n] = 0
            x_posish[n] = 0
            y_posish[n] = 0

            n = n + 1
            continue

        img_filt = ndimage.median_filter(img, 3)

        start_vals = [(np.mean(img_filt), p_contrasts[n][max_i,4], c_pos[0,n], c_pos[1,n], p_contrasts[n][max_i,3])]

        c_gaus[n], x_posish[n], y_posish[n] = gausfit_2d(img_filt, start_vals, False, path, fn, n)

        n = n + 1

    c_all = np.vstack((c_ind, x_posish, y_posish, c_max, c_gaus)) 
    c_all = c_all[:, c_all[0, :].argsort()]

    if view_values['-track_opt-'] == 'Tracking On':

        c_filt = np.empty((1,5))

        d_f = 30
        d_px = 15

        row = c_all[:,0]

        (frame, pos_x, pos_y,_,_) = row

        c_temp = np.reshape(row, (-1, 5))

        for i in range(1,np.shape(c_all)[1]):
            
            comp_row = c_all[:,i]

            if comp_row[0] - frame < d_f and abs(comp_row[1] - pos_x) < d_px and abs(comp_row[2] - pos_y) < d_px:

                (frame, pos_x, pos_y,_,_) = comp_row

                c_temp = np.append(c_temp, np.reshape(comp_row, (-1, 5)), axis = 0)

            else:
                print('--------')
                print(c_temp)

                c_filt = np.append(c_filt, np.reshape( c_temp[np.argmax(abs(c_temp[:,4])), :], (-1, 5) ), axis = 0)

                row = comp_row

                (frame, pos_x, pos_y,_,_) = row

                c_temp = np.reshape(row, (-1, 5))

        c_filt = np.append(c_filt, np.reshape( c_temp[np.argmax(abs(c_temp[:,4])), :], (-1, 5) ), axis = 0)
        c_all = np.delete(c_filt, 0, 0)
        
    pd.DataFrame(c_all).to_csv(os.path.join(path, fn)+'_contrasts.txt', header=None, index=None)
    np.save((os.path.join(path, fn)+'_contrasts'), c_all)

# %%