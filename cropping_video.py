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


    # Show the selected video and enable a rectangular selection for the section which one wants to crop out 
    def select_roi(self, frame_index=10):
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
    

    # Crop the selected section out of the video
    def crop_video(self):
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


    # Generate output path, which is the same folder as the original video was saved in. The cropped
    # video has the same name as the original, but with the surfix _cropped_number.
    def generate_output_path(self):
        base_name, ext = os.path.splitext(self.video_path)
        dir_name = os.path.dirname(base_name)
        file_name = os.path.basename(base_name)

        # Find all files that match the pattern
        existing_files = [f for f in os.listdir(dir_name) if f.startswith(file_name + "_cropped") and f.endswith(ext)]

        # Extract the highest enumeration number
        max_counter = 0
        for f in existing_files:
            try:
                num_str = f.replace(file_name + "_cropped_", "").replace(ext, "")
                num = int(num_str)
                if num > max_counter:
                    max_counter = num
            except ValueError:
                continue

        # Generate the new file name
        new_counter = max_counter + 1
        new_path = os.path.join(dir_name, f"{file_name}_cropped_{new_counter}{ext}")
        return new_path


    # Save the cropped video
    def save_cropped_video(self, cropped_video_data):
        output_path = self.generate_output_path()
        np.save(output_path, cropped_video_data)
        print(f"Cropped video saved to {output_path}")

