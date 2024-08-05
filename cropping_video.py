import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import os
import PySimpleGUI as sg

class VideoCropper:

    def __init__(self, video_path):
        self.video_path = video_path
        self.video_data = np.load(video_path)
        self.roi = None

    def select_roi(self, frame_index=10):
        """
        Display a frame to select ROI and store the selected ROI.
        """
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            plt.close()

        frame = self.video_data[frame_index]
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap='gray')
        rect_selector = RectangleSelector(ax, onselect, 
                                          button=[1], minspanx=5, minspany=5, 
                                          spancoords='pixels', interactive=True)
        plt.title("Select ROI and close the window")
        plt.show()

        if self.roi is None:
            raise ValueError("ROI not selected.")
    
    def crop_video(self):
        """
        Crop the video based on the selected ROI.
        """
        if self.roi is None:
            raise ValueError("ROI must be selected before cropping.")

        x, y, w, h = self.roi
        num_frames = self.video_data.shape[0]
        cropped_video_data = np.zeros((num_frames, h, w), dtype=self.video_data.dtype)

        for frame_index in range(num_frames):
            frame = self.video_data[frame_index]
            cropped_frame = frame[y:y+h, x:x+w]
            cropped_video_data[frame_index] = cropped_frame
        
        return cropped_video_data

    def save_cropped_video(self, cropped_video_data, output_path):
        """
        Save the cropped video data to a .npy file.
        """
        np.save(output_path, cropped_video_data)
        print(f"Cropped video saved to {output_path}")

