#function to generate a linear calibration of the detector using a Eu-152 Spectrum
#Author: Thomas Bailey
#Date Created: 20181114

import numpy as np
import pandas as pd
import glob
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import uncertainties as u

#fit a peak to a region of data with width fit_width around the peak bin
def peak_fit(data, peak, fit_width):
    fit_bins = data.loc[(data['BinNumber']>(peak-fit_width/2))].loc[data['BinNumber']<(peak+fit_width/2)]
    counts = np.log(fit_bins['Counts'].astype(int)+1) #take log so that peaks are more prominent, +1 to prevent error with 0 entries
    bins = fit_bins['BinNumber'].astype(int)
    popt, pcov = curve_fit(peak_function, bins, counts, p0=[peak, 1, 6, 1], bounds=([peak-20, 0.01, 1, 0.5], [peak+20, 3, 15, 10]))
    return popt, np.sqrt(np.diag(pcov))

#function used as the fitted calibration
#in this order so that bin peak can be on the y-axis during fit
def calibration_fit_function(energy, a, b):
    return (energy - b) / a

def Calibration_Fitter(filename):
    #load data into a panda array
    df, metadata = h5load(filename)

    #list to store peak centers in
    peak_bins = []
    peak_bin_uncertainties = []

    #plt.plot(df['BinNumber'], np.log(df['Counts'].astype(int)+1), 'x')

    known_energies = [112.78, 244.7, 344.28, 411.12, 443.96, 778.9, 867.37, 964.08, 1085.9, 1112.1, 1299.1, 1408]

    #perform a fit for each peak to find the central value
    for peak in known_energies:
        fit_values, fit_errors = peak_fit(df, peak, 60)
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
    
    plt.savefig('Calibration_line'+str(metadata['RealTime'])+'.pdf')
    plt.close()
    return u.ufloat(lin_fit[0], np.sqrt(np.diag(pcov))[0]), u.ufloat(lin_fit[1], np.sqrt(np.diag(pcov))[1]), metadata['TimeStamp']

a_list = []
b_list = []
time_list = []

for file in glob.iglob('C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject/Data/Experimental/*Eu_calibration.h5'):
    print("Calibrating form file: {}".format(file))
    a, b, time = Calibration_Fitter(file)
    data, metadata = h5load(file)
    a_list.append(a)
    b_list.append(b)
    time = time[6:10]+time[0:2]+time[3:5]+time[11:13]+time[14:16]+time[17:-1]
    time = int(time)
    time -= 2018110611145 #remove offset from not being in year 0
    time_list.append(time)

#save calibrations to a file
out_file = open("Calibrations.csv", "w")
out_file.write("Time Stamp, a, b\n")
for i in range(len(a_list)):
    out_file.write(str(time_list[i]+2018110611145)+', '+str(a_list[i])+', '+str(b_list[i])+'\n')
out_file.close()

mean_a = u.ufloat(np.mean([i.n for i in a_list]), np.std([i.n for i in a_list]))
mean_b = u.ufloat(np.mean([i.n for i in b_list]), np.std([i.n for i in b_list]))
print(mean_a)
print(mean_b)

#plot the gradient of the fit against time
plt.errorbar(time_list, [i.n for i in a_list], [i.s for i in a_list], fmt='x', color='black', linestyle='')
plt.plot([0,700000],[mean_a.n, mean_a.n], 'r')
plt.plot([0,700000],[mean_a.n+mean_a.s, mean_a.n+mean_a.s], linestyle='--', color='red', alpha=0.5)
plt.plot([0,700000],[mean_a.n-mean_a.s, mean_a.n-mean_a.s], linestyle='--', color='red', alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Gradient of Calibration (KeV/bin)")
plt.savefig('a_fit.pdf')
plt.close()

#plot the gradient of the intercept of the fit against time
plt.errorbar(time_list, [i.n for i in b_list], [i.s for i in b_list], fmt='x', color='black', linestyle='')
plt.plot([0,700000],[mean_b.n, mean_b.n], 'r')
plt.plot([0,700000],[mean_b.n+mean_b.s, mean_b.n+mean_b.s], linestyle='--', color='red', alpha=0.5)
plt.plot([0,700000],[mean_b.n-mean_b.s, mean_b.n-mean_b.s], linestyle='--', color='red', alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Intercept of Calibration (KeV)")
plt.savefig('b_fit.pdf')
