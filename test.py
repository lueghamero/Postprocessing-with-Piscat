import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import tkinter as tk

filename = sg.popup_get_file('Filename to play')

video = np.load(filename)

batchSize = 1
size_A_diff = (video.shape[0] - 2 * batchSize, video.shape[1], video.shape[2])
output_batch_1 = np.empty(size_A_diff)
output_diff = np.empty(size_A_diff)

batch_1 = np.sum(video[0 : batchSize, :, :], axis=0)
batch_2 = np.sum(video[batchSize : 2 * batchSize, :, :], axis=0)

batch_1_ = np.divide(batch_1, batchSize)
batch_2_ = np.divide(batch_2, batchSize)

output_diff[0, :, :] = batch_2_ - batch_1_
output_batch_1[0, :, :] = batch_1_

for i_ in range(1, video.shape[0] - 2 * batchSize):
    batch_1 = (batch_1 - video[i_ - 1, :, :] + video[batchSize + i_ - 1, :, :])
    batch_2 = (
            batch_2
            - video[batchSize + i_ - 1, :, :]
            + video[(2 * batchSize) + i_ - 1, :, :]
            )
    batch_1_ = np.divide(batch_1, batchSize)
    batch_2_ = np.divide(batch_2, batchSize)

    output_diff[i_, :, :] = batch_2_ - batch_1_
    output_batch_1[i_, :, :] = batch_1_
    plt.imshow(output_diff[i_])
    plt.show()
    plt.close()

