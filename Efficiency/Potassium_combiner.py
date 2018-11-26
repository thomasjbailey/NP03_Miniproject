#a script to combine all the low sodium salt spectra into one data file
#Author: Thomas Bailey
#Date Created: 20181126

import numpy as np
import pandas as pd
import glob
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as u
import pickle


#energy calibration functions
def calibrate_array(data, a, b):
    return a * data + b

calibrated_data = []
#loop to open, calibrate each file using metadata calibration and then add to unbinned combined set
for file in glob.iglob("../Data/Experimental/LowSodium*.h5"):
    data, metadata = h5load(file)
    data['Energy'] = data['BinNumber'].apply(calibrate_array, args=tuple(metadata['Calibration']))
    calibrated_data += unbin(data)

plt.hist(calibrated_data, 1000, log=True)
plt.show()

#save combined unbinned uranium data
with open('Potassium_data.txt', 'wb') as fp:
    pickle.dump(calibrated_data, fp)
