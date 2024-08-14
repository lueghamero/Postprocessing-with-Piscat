# Display the PSFs

import numpy as np
from piscat.Visualization import * 
from piscat.Localization import *
import pandas as pd
import dill as pickle  # Use dill instead of pickle

# loading the video
video_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/DRAd_Bio.npy'
dra_video = np.load(video_path)

# loading the df as pickleable
psf_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/df_PSF.pkl'
with open(psf_path, 'rb') as file:
    df_PSFs = pickle.load(file)

# where the video will be saved 
save_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/gif2.gif'
display_psf = DisplayDataFramePSFsLocalization(dra_video, df_PSFs, 0.1, False , save_path)

# to make the DisplayDataFramePSFsLocalization pickleable 

# Extract only the pickleable data
picklable_data = {
    'df_PSFs': display_psf.df_PSFs,
    'dra_video': display_psf.video
}

# Save the pickleable data
pickle_path = '/Users/ipeks/Desktop/DNA_PAINT_ISCAT/iScatData/display_psf_data.pkl'
with open(pickle_path, 'wb') as file:
    pickle.dump(picklable_data, file)

# Load the pickleable data
with open(pickle_path, 'rb') as file:
    data_loaded = pickle.load(file)

# Reconstruct the object (you may need to reinitialize some non-pickleable parts)
display_psf_loaded = DisplayDataFramePSFsLocalization( data_loaded['dra_video'], data_loaded['df_PSFs'],  0.1, False ,save_path )


# Now you can use the loaded object
display_psf_loaded.run()