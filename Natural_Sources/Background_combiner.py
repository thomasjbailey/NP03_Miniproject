#a script to combine all the background spectra into one data file
#Author: Thomas Bailey
#Date Created: 20181127

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

#function used as the fitted calibration
#in this order so that bin peak can be on the y-axis during fit
def calibration_fit_function(energy, a, b):
    return (energy - b) / a

#energy calibration functions
def calibrate_array(data, a, b):
    return a * data + b

#fit a peak to a region of data with width fit_width around the peak bin
def peak_fit(data, peak, fit_width):
    fit_bins = data.loc[(data['BinNumber']>(peak-fit_width/2))].loc[data['BinNumber']<(peak+fit_width/2)]
    counts = np.log(fit_bins['Counts'].astype(int)+1) #take log so that peaks are more prominent, +1 to prevent error with 0 entries
    bins = fit_bins['BinNumber'].astype(int)
    popt, pcov = curve_fit(peak_function, bins, counts, p0=[peak, 1, 6, 1], bounds=([peak-20, 0.01, 1, 0.5], [peak+20, 3, 15, 10]))
    return popt, np.diag(pcov)

def Calibrate(df, metadata):
    #list to store peak centers in
    peak_bins = []
    peak_bin_uncertainties = []

    plt.plot(df['BinNumber'], np.log(df['Counts'].astype(int)+1), 'x')

    known_energies = [75, 84.8, 511, 609.31, 661.66, 1120.29, 1460.8, 1588.20, 1764.49]
    #perform a fit for each peak to find the central value
    for peak in known_energies:
        #fit peak around bin number expected from meta data calibration
        fit_values, fit_errors = peak_fit(df, (peak-metadata['Calibration'][1])/metadata['Calibration'][0], 30)
        peak_bins.append(fit_values[0])
        peak_bin_uncertainties.append(np.sqrt(fit_errors[0]))
        plt.plot(np.linspace(peak-25, peak+25), peak_function(np.linspace(peak-25, peak+25), *fit_values))
    plt.show()

    lin_fit, pcov = curve_fit(calibration_fit_function, known_energies, peak_bins, sigma=peak_bin_uncertainties)
    df['Energy'] = df['BinNumber'].apply(calibrate_array, args=tuple(lin_fit))
    return df

calibrated_data = []
hists = []
#loop to open, calibrate each file and then add to unbinned combined set
for file in glob.iglob("../Data/Experimental/*Background*.h5"):
    data, metadata = h5load(file)
    #print(metadata)
    print(u.ufloat(data['Counts'].sum(), np.sqrt(data['Counts'].sum())) / metadata['RealTime'])
    data = Calibrate(data, metadata)
    calibrated_data.append(unbin(data))
    binned_background, edges2 = np.histogram(unbin(data), 200)
    binned_background = binned_background / metadata['RealTime']
    hists.append(binned_background)


#data['Energy'] = data['BinNumber'].apply(Calibrate, args=tuple(metadata['Calibration']))
plt.plot([i for i in range(200)], hists[0], alpha = 0.5)
plt.plot([i for i in range(200)], hists[1], alpha = 0.5)
plt.plot([i for i in range(200)], hists[2], alpha = 0.5)
plt.yscale('log')
plt.show()

#save combined unbinned background data
#with open('Background_data.txt', 'wb') as fp:
#    pickle.dump(calibrated_data, fp)
