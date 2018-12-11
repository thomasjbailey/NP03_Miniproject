#a script to combine all the uranium spectra into one data file
#Author: Thomas Bailey
#Date Created: 20181118

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

def Calibrate(df):
    #list to store peak centers in
    peak_bins = []
    peak_bin_uncertainties = []

    #plt.plot(df['BinNumber'], np.log(df['Counts'].astype(int)+1), 'x')

    known_energies = [129.1, 209.4, 238.6, 338.4, 463, 511.0, 583.1, 911.1]
    #perform a fit for each peak to find the central value
    for peak in known_energies:
        fit_values, fit_errors = peak_fit(df, peak, 30)
        peak_bins.append(fit_values[0])
        peak_bin_uncertainties.append(fit_errors[0])
        #plt.plot(np.linspace(peak-25, peak+25), peak_function(np.linspace(peak-25, peak+25), *fit_values))
    #plt.show()

    lin_fit, pcov = curve_fit(calibration_fit_function, known_energies, peak_bins, sigma=peak_bin_uncertainties)

    f, axes = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[4, 1], 'hspace': 0.03})

    axes[0].plot([0, 1500], [lin_fit[1], 1500*lin_fit[0]+lin_fit[1]], '--r',)
    axes[0].errorbar(peak_bins, known_energies, peak_bin_uncertainties, color = 'black', linestyle='none', marker = 'x')
    axes[0].set_ylabel('Energy [KeV]')
    axes[0].set_xlim(0, 1500)
    axes[0].set_ylim(0, 1500)

    difference = lin_fit[0]*np.array(peak_bins)+lin_fit[1] - np.array(known_energies)
    axes[1].plot(peak_bins, difference, 'x', color='gray')
    axes[1].plot([0,1500], [0,0], color='black', linewidth=0.5)
    axes[1].set_xlim(0, 1500)

    axes[1].set_xlabel('Bin Number')
    axes[1].set_ylabel('Residual')

    plt.show()

    df['Energy'] = df['BinNumber'].apply(calibrate_array, args=tuple(lin_fit))
    return df

calibrated_data = []
#loop to open, calibrate each file and then add to unbinned combined set
for file in glob.iglob("../Data/Experimental/232-Th*.h5"):
    data, metadata = h5load(file)
    data = Calibrate(data)
    calibrated_data += unbin(data)

#data['Energy'] = data['BinNumber'].apply(Calibrate, args=tuple(metadata['Calibration']))
plt.hist(calibrated_data, 1000, log=True)
plt.show()

#save combined unbinned uranium data
with open('Thorium_data.txt', 'wb') as fp:
    pickle.dump(calibrated_data, fp)
