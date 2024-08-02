    # -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:07:39 2023

@author:clara from github
"""

#%% 
from copyreg import dispatch_table

import PySimpleGUI as sg
import numpy as np
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skimage
from scipy.optimize import curve_fit
import sys
import os

# --- General --- #

#: Use to draw a live figure with PysimpleGUI
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure,canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def get_scaling():
    root = sg.tk.Tk()
    scaling = root.winfo_fpixels('1i')/72
    root.destroy()
    return scaling

#: Make a folder if it does not exits yet
def make_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)

#: Display a progress bar
def progressbar(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush() 

##: load a mat file 
# def load_mat(filename, varname):
#     try:
#         vidFile = np.squeeze(list(sio.loadmat(filename).values())[3])
#     except:
#         vidFile_dict = mat73.loadmat(filename)
#         vidFile = vidFile_dict[varname]  

#     return vidFile

def load_npy(file_path):
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

def vid_norm(video):
    vid_norm = np.sum(video, (0,1))/(np.shape(video)[0]*np.shape(video)[1])
    vid = video/vid_norm

    return vid


# --- Image Processing --- #

#: Subtract a Noise Map from the current Image
#Options: 'Noise Map: Rows', 'Noise Map: Columns' or 'Noise Map: Both' - otherwise none
def noise_map(img, opt):
    median_col = np.median(img, axis = 0)
    median_row = np.median(img, axis = 1)
    np_col_y, np_row_y = np.meshgrid(median_col, median_row)

    if opt == 'Noise Map: Rows':
        img_np = img - np_row_y
    elif opt == 'Noise Map: Columns':
        img_np = img - np_col_y
    elif opt == 'Noise Map: Both':
        img_np = img - np_row_y - np_col_y
    else:
        img_np = img

    return img_np

#: Fourier Filter - set a threshold and the radius of low and high spatial filter
def ft_filter(img, gfilter_low, gfilter_high, threshold):

    # FFT
    img_ft = np.fft.fftshift(np.fft.fft2(img))
        
    # Filter

    img_ft = img_ft*(np.abs(img_ft) < threshold)

    img_apt = np.ones(np.shape(img_ft))
    x = np.shape(img_ft)[0]
    y = np.shape(img_ft)[1]
    X, Y = np.meshgrid(np.linspace(int(-x/2), int(x/2), x), np.linspace(int(-y/2), int(y/2), y)) 

    img_apt = img_apt*(np.exp(-X**2/gfilter_high**2-Y**2/gfilter_high**2))
    img_apt = img_apt*(1-np.exp(-X**2/gfilter_low**2-Y**2/gfilter_low**2))
    img_ft_filt = img_ft*img_apt

    # Rev FFT
    img_rev_ft = np.fft.ifft2(np.fft.ifftshift(img_ft_filt))
    img_rev_ft = np.real(img_rev_ft)

    return img_ft, img_ft_filt, img_rev_ft

# --- View Image --- #

#: Initilaze View Image
def view_ini(vid, avg, i, view_opt, map_opt):

    if view_opt == 'Differential':
        disp = np.sum(vid[:,:,i:i+avg], 2)-np.sum(vid[:,:,i+avg:i+2*avg], 2)

    elif view_opt == 'Mean Filter':
        disp = vid[:,:,i+avg] - np.sum(vid[:,:,i:i+2*avg], 2)/(2*avg)

    else: 
        disp = np.mean(vid[:,:,i:i+avg], 2)

    disp = noise_map(disp, map_opt)
    
    return disp


def view_cont(disp, vid, avg, i, view_opt, map_opt): 

    if view_opt == 'Differential':
        disp = disp - vid[:,:,i-1] + 2*vid[:,:,i+avg-1] - vid[:,:,i+2*avg-1]

    elif view_opt == 'Mean Filter':
        disp = disp - vid[:,:,i-1+avg] + vid[:,:,i+avg] + vid[:,:,i-1]/(2*avg) - vid[:,:,i+2*avg]/(2*avg)

    else: 
        disp = np.mean(vid[:,:,i:i+avg], 2)

    disp = noise_map(disp, map_opt)
    
    return disp

# --- Particle Detection --- #

#: Find Blobs in an Image using skimage
def find_blobs(img, th, i):

    dog_blobs_p = skimage.feature.blob_dog(img, threshold = th, min_sigma = 1, max_sigma = 3)
    dog_blobs_n = skimage.feature.blob_dog(img*-1, threshold = th, min_sigma = 1, max_sigma = 3)
  
    dog_blobs = np.concatenate((dog_blobs_p, dog_blobs_n), axis = 0)

    dog_blobs = np.delete(dog_blobs, np.where((dog_blobs[:,0:2] < 5) | (dog_blobs[:,0:2] > np.amin(np.shape(img))-5))[0], axis = 0)

    dog_blobs = np.insert(dog_blobs, 0, i, axis = 1)
    dog_blobs = np.insert(dog_blobs, 4, 0, axis = 1)

    for row in range(0,np.shape(dog_blobs)[0]):
        x = int(dog_blobs[row,1])
        y = int(dog_blobs[row,2])
        dog_blobs[row,4] = img[x,y]


    return dog_blobs

#: Group Particles properly and remove duplicates
def group_p(particles):

    all_p = np.unique(particles[:,1:3], axis = 0)

    if np.shape(all_p[:,0])[0] > np.shape(np.unique(all_p[:,0]))[0] or np.shape(all_p[:,0])[0] > np.shape(np.unique(all_p[:,1]))[0]:

        for row in range(0,np.shape(all_p)[0]): 
            all_p[:,0] = np.where(abs(all_p[:,0]-all_p[row,0]) > 5, all_p[:,0], all_p[row,0])
            all_p[:,1] = np.where(abs(all_p[:,1]-all_p[row,1]) > 5, all_p[:,1], all_p[row,1])

        all_p = np.unique(all_p, axis = 0)

    return all_p

#: 2D Gaus Function
def gaus_2d(X, Y, b, A, xc, yc, sig):

    return b + A * np.exp(-(((X - xc) ** 2 / ((sig ** 2))) + ((Y - yc) ** 2 / ((sig ** 2)))))

#: Helper Function for Gausfit
def _gaus(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
       arr += gaus_2d(x, y, *args[i*5:i*5+5])
    return arr

#: 2D Gausfit
def gausfit_2d(Z, start_vals, save_plot, path, fn, n):
                
    x = np.linspace(0, np.shape(Z)[0] - 1, np.shape(Z)[0], dtype=int)
    y = np.linspace(0, np.shape(Z)[1] - 1, np.shape(Z)[1], dtype=int)
    X, Y = np.meshgrid(x, y) 

    guess_prms = start_vals
    p0 = [p for prms in guess_prms for p in prms]

    xdata = np.vstack((X.ravel(), Y.ravel()))

    popt, pcov = curve_fit(_gaus, xdata, Z.ravel(), p0)

    fit = np.zeros(Z.shape)
    fit = gaus_2d(X, Y, *popt)

    # fig, ax = plt.subplot_mosaic([
    #     ['img', 'fit']
    # ], figsize=(7, 3.5))

    # ax['img'].imshow(Z, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    # ax['img'].set_title('Image')

    # ax['fit'].imshow(Z, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
    # ax['fit'].contour(X, Y, fit, 3,colors='r')
    # ax['fit'].set_title('Fit')

    # if save_plot == True:

    #     fn = fn + '_gausfit_' + str(n)

    #     if not os.path.exists(os.path.join(path, fn)+'.png'):
    #         fig.savefig(os.path.join(path, fn)+'.png')

   # plt.show()
    

    return popt[1], popt[2], popt[3]
# %%