import os
import warnings

import pandas as pd
import math
import numpy as np
from PySide6.QtCore import Slot

from skimage import feature
from skimage.feature import peak_local_max
from scipy import optimize

import matplotlib.patches as patches

from piscat.Localization import data_handling, gaussian_2D_fit, radial_symmetry_centering
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.Preproccessing import filtering

from scipy.spatial.distance import pdist, squareform


class PSFsExtractionPreview:
    def __init__(self, video_shape, flag_transform=False, **kwargs):
        """Adapted to process one frame at a time."""
        super(PSFsExtractionPreview, self).__init__()
        self.kwargs = kwargs
        self.counter = 0

        # Placeholder for the frame shape instead of the entire video
        self.video_shape = video_shape
        self.flag_transform = flag_transform

        self.psf_dog = None
        self.psf_doh = None
        self.psf_log = None
        self.psf_hog = None
        self.psf_frst = None

        self.psf_hog_1D_feature = None

        self.min_sigma = None
        self.max_sigma = None
        self.sigma_ratio = None
        self.threshold = None
        self.overlap = None
        self.radii = None
        self.alpha = None
        self.beta = None
        self.stdFactor = None
        self.mode = None
        self.function = None

        self.df_PSF = None
        self.min_distance = None



    @Slot()
    def run(self):
        result = self.psf_detection(**self.kwargs)
        self.signals.result.emit(result)

    def dog(self, image):

        return feature.blob_dog(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            sigma_ratio=self.sigma_ratio,
            threshold=self.threshold,
            overlap=self.overlap,
            exclude_border=True,
        )

    def doh(self, image):

        tmp = feature.blob_doh(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=int(self.sigma_ratio),
            threshold=self.threshold,
            overlap=self.overlap,
        )
        return tmp

    def log(self, image):

        return feature.blob_log(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=int(self.sigma_ratio),
            threshold=self.threshold,
            overlap=self.overlap,
            exclude_border=True,
        )

    def _rvt(self, image):

        if self.flag_transform:
            tr_img = image
        else:
            rvt_ = filtering.RadialVarianceTransform()
            tr_img = rvt_.rvt(
                img=image,
                rmin=self.min_radial,
                rmax=self.max_radial,
                kind=self.rvt_kind,
                highpass_size=self.highpass_size,
                upsample=self.upsample,
                rweights=self.rweights,
                coarse_factor=self.coarse_factor,
                coarse_mode=self.coarse_mode,
                pad_mode=self.pad_mode,
            )

        local_maxima = peak_local_max(
            tr_img,
            threshold_abs=self.threshold,
            footprint=np.ones((3,) * (image.ndim)),
            threshold_rel=0.0,
            exclude_border=True,
        )

        # Catch no peaks
        if local_maxima.size == 0:
            return np.empty((0, 3))

        sigmas = (self.min_radial / np.sqrt(2)) * np.ones((local_maxima.shape[0], 1))
        tmp = np.concatenate((local_maxima, sigmas), axis=1)

        return tmp

    def psf_detection(self, frame, frame_index, function, **kwargs):
        """Process a single frame instead of the entire video."""
        self.min_sigma = kwargs.get('min_sigma', 1)
        self.max_sigma = kwargs.get('max_sigma', 8)
        self.sigma_ratio = kwargs.get('sigma_ratio', 1.1)
        self.threshold = kwargs.get('threshold', 0.005)
        self.overlap = kwargs.get('overlap', 0)
        self.min_radial = kwargs.get('min_radial', 1)
        self.max_radial = kwargs.get('max_radial', 2)
        self.radial_step = kwargs.get('radial_step', 0.1)
        self.alpha = kwargs.get('alpha', 2)
        self.beta = kwargs.get('beta', 1)
        self.stdFactor = kwargs.get('stdFactor', 1)
        self.mode = kwargs.get('mode', "BOTH")
        self.min_distance = kwargs.get("min_distance", 15)
        self.function = function

        # Detect PSFs in the current frame
        result = self.psf_detection_kernel(frame, frame_index)
        return result

    def psf_detection_kernel(self, frame, frame_index):
        """Process a single frame and apply the specified function."""
        if self.function == "dog":
            if self.mode == "BOTH":
                positive_psf = self.dog(frame)
                negative_psf = self.dog(-1 * frame)
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)
            elif self.mode == "Bright":
                positive_psf = self.dog(frame)
                negative_psf = []
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)
            elif self.mode == "Dark":
                positive_psf = []
                negative_psf = self.dog(-1 * frame)
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)

        elif self.function == "doh":
            if self.mode == "BOTH":
                positive_psf = self.doh(frame)
                negative_psf = []
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)

        elif self.function == "log":
            if self.mode == "BOTH":
                positive_psf = self.log(frame)
                negative_psf = self.log(-1 * frame)
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)
            elif self.mode == "Bright":
                positive_psf = self.log(frame)
                negative_psf = []
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)
            elif self.mode == "Dark":
                positive_psf = []
                negative_psf = self.log(-1 * frame)
                temp2 = self.concatenateBrightDark(positive_psf, negative_psf, frame_index, frame)

        elif self.function == "frst_one_psf":
            b_psf = self.frst_one_PSF(frame)
            temp2 = self.concatenateBrightDark(b_psf, [], frame_index, frame)
            temp2 = np.expand_dims(temp2, axis=0)

        elif self.function == "RVT":
            b_psf = self._rvt(frame)
            temp2 = self.concatenateBrightDark(b_psf, [], frame_index, frame)
            temp2 = np.expand_dims(temp2, axis=0)

        if temp2 is not None:
            temp2 = self.filter_adjacent_blobs(temp2)
                
        return temp2
    

    def concatenateBrightDark(self, bright_psf, dark_psf, frame_index, frame):
        """Concatenate bright and dark PSFs into a single structure, and add intensity as the 5th entry."""
        def calculate_intensity(psf, frame):
            """Calculate intensity as the pixel value at the PSF's location (y, x)."""
            # Extract intensity based on the center of the PSF (y, x coordinates)
            y, x = int(psf[0]), int(psf[1])
            return frame[y, x]  # Use frame's pixel value as intensity
        
        # Calculate intensity for bright PSFs
        if len(bright_psf) != 0:
            bright_psf_with_intensity = np.hstack([bright_psf, np.array([[calculate_intensity(psf, frame)] for psf in bright_psf])])
        else:
            bright_psf_with_intensity = []

        # Calculate intensity for dark PSFs
        if len(dark_psf) != 0:
            dark_psf_with_intensity = np.hstack([dark_psf, np.array([[calculate_intensity(psf, frame)] for psf in dark_psf])])
        else:
            dark_psf_with_intensity = []
        
        # Concatenate bright and dark PSFs with intensity
        if len(bright_psf_with_intensity) != 0 and len(dark_psf_with_intensity) != 0:
            psf = np.unique(np.concatenate((bright_psf_with_intensity, dark_psf_with_intensity)), axis=0)
            frame_num = frame_index * np.ones((psf.shape[0], 1), dtype=int)
            temp2 = np.concatenate((frame_num, psf), axis=1)
        elif len(bright_psf_with_intensity) != 0:
            psf = np.asarray(bright_psf_with_intensity)
            frame_num = frame_index * np.ones((psf.shape[0], 1), dtype=int)
            temp2 = np.concatenate((frame_num, psf), axis=1) if psf.ndim > 1 else np.concatenate(([frame_index], psf), axis=0)
        elif len(dark_psf_with_intensity) != 0:
            psf = np.asarray(dark_psf_with_intensity)
            frame_num = frame_index * np.ones((psf.shape[0], 1), dtype=int)
            temp2 = np.concatenate((frame_num, psf), axis=1)
        else:
            temp2 = None

        return temp2
    

    def filter_adjacent_blobs(self, blobs, min_distance=None):

        if blobs is None or len(blobs) == 0:
            return blobs

        # Use the class-defined min_distance if not provided
        if min_distance is None:
            min_distance = self.min_distance

        # Extract the coordinates and intensity values
        coordinates = blobs[:, 1:3]  # Assuming y, x are at columns 1 and 2
        intensities = blobs[:, 4]    # Assuming intensity is at column 4

        # Compute pairwise distances between blobs
        distances = squareform(pdist(coordinates))

        # Sort the blobs by intensity (descending order)
        sorted_indices = np.argsort(np.abs(intensities))[::-1]
        sorted_blobs = blobs[sorted_indices]

        # Initialize a mask to keep track of blobs to remove
        keep_mask = np.ones(len(sorted_blobs), dtype=bool)

        # Iterate over each pair of blobs, preferring the higher intensity (already sorted)
        for i in range(len(sorted_blobs)):
            if not keep_mask[i]:  # If already marked for removal, skip
                continue
            for j in range(i + 1, len(sorted_blobs)):
                if distances[sorted_indices[i], sorted_indices[j]] < min_distance:
                    # Mark the lower intensity blob for removal
                    keep_mask[j] = False

        # Keep only the blobs that passed the filtering
        filtered_blobs = sorted_blobs[keep_mask]

        return filtered_blobs
    
    
    def create_red_circles(self, psf_positions, ax, radius=5):

        circles = []
        for psf in psf_positions:
            # PSF positions are assumed to be in the format [frame_num, y, x, sigma]
            y, x = psf[1], psf[2]
            circle = patches.Circle((x, y), radius=radius, edgecolor='red', facecolor='none', lw=1)
            ax.add_patch(circle)
            circles.append(circle)
        
        return circles
    
    def get_psf_image(self, frame, position, size=15):
        """
        Extracts a square region of the image around a PSF position.

        Parameters
        ----------
        frame: np.ndarray
            The image frame.

        position: tuple
            The (y, x) position of the PSF.

        size: int
            The size of the extracted region.

        Returns
        -------
        np.ndarray
            The extracted image region.
        """
        y, x = position
        half_size = size // 2
        
        # Ensure the PSF image stays within the frame boundaries
        y_min = max(int(y - half_size), 0)
        y_max = min(int(y + half_size), frame.shape[1])
        x_min = max(int(x - half_size), 0)
        x_max = min(int(x + half_size), frame.shape[0])
        
        # Return both the PSF image and the (y_min, x_min) corner
        return frame[y_min:y_max, x_min:x_max], (y_min, x_min)