#a scipt to compare different spectra with background
#Author: Thomas Bailey
#date created: 20190103

import numpy as np
import pandas as pd
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as u
import pickle
import iminuit, probfit
import scipy.stats
from joblib import Parallel, delayed
import scipy.constants as c
import glob

#energy calibration functions
def calibrate_array(data, a, b):
    return a * data + b

# #get values of counts in each peak for background
data_file = open("../Natural_Sources/Background_data.txt", 'rb')
print('Loading Background Data from file')
Background_data = pickle.load(data_file)
data_file.close()

#get unbinned data for thorium
data_file = open("Thorium_data.txt", 'rb')
print('Loading Thorium Data from file')
Thorium_data = pickle.load(data_file)
data_file.close()

#get unbinned data for uranium
data_file = open("Uranium_data.txt", 'rb')
print('Loading Uranium Data from file')
Uranium_data = pickle.load(data_file)
data_file.close()

#get unbinned data for brazil nuts
data_file = open("../Natural_Sources/BrazilNut_data.txt", 'rb')
print('Loading Brazil Nut Data from file')
BrazilNut_data = pickle.load(data_file)
data_file.close()


binned_thorium, edges = np.histogram(Thorium_data, 200)
thorium_uncertainty = np.sqrt(binned_thorium)
binned_thorium = binned_thorium / (2132+4975+417)
thorium_uncertainty = thorium_uncertainty / (2132+4975+417)
centers = [(edges[i]+edges[i+1])/2 for i in range(len(binned_thorium))]
w = (centers[1]-centers[0])
binned_thorium = binned_thorium / w
thorium_uncertainty = thorium_uncertainty / w
plt.bar(centers, binned_thorium, width = w, alpha =0.6, align='center', label='Thorium', yerr=thorium_uncertainty, color='C0', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C0', 'capsize': 2})

binned_uranium, edges3 = np.histogram(Uranium_data, 200)
uranium_uncertainty = np.sqrt(binned_uranium)
binned_uranium = binned_uranium / (2164+1644+3640+3902)
uranium_uncertainty = uranium_uncertainty / (2164+1644+3640+3902)
centers3 = [(edges3[i]+edges3[i+1])/2 for i in range(len(binned_uranium))]
w3 = (centers3[1]-centers3[0])
binned_uranium = binned_uranium / w3
uranium_uncertainty = uranium_uncertainty / w3
plt.bar(centers3, binned_uranium, width = w3, alpha =0.6, align='center', label='Uranium', yerr=uranium_uncertainty, color='C2', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C2', 'capsize': 2})

binned_brazilnut, edges4 = np.histogram(BrazilNut_data, 200)
brazilnut_uncertainty = np.sqrt(binned_brazilnut)
binned_brazilnut = binned_brazilnut / (63199)
brazilnut_uncertainty = brazilnut_uncertainty / (63199)
centers4 = [(edges4[i]+edges4[i+1])/2 for i in range(len(binned_brazilnut))]
w4 = (centers4[1]-centers4[0])
binned_brazilnut = binned_brazilnut / w4
brazilnut_uncertainty = brazilnut_uncertainty / w4
plt.bar(centers4, binned_brazilnut, width = w4, alpha =0.6, align='center', label='Brazil Nut', yerr=brazilnut_uncertainty, color='C3', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C3', 'capsize': 2})

binned_background, edges2 = np.histogram(Background_data, 200)
background_uncertainty = np.sqrt(binned_background)
binned_background = binned_background / (164277+242316)
background_uncertainty = background_uncertainty / (164277+242316)
centers2 = [(edges2[i]+edges2[i+1])/2 for i in range(len(binned_background))]
w2 = (centers2[1]-centers2[0])
binned_background = binned_background / w2
background_uncertainty = background_uncertainty / w2
plt.bar(centers2, binned_background, width = w2, alpha =0.6, align='center', label='Background', yerr=background_uncertainty, color='C1', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C1', 'capsize': 2})

plt.xlabel('Energy [KeV]')
plt.ylabel(r'Count Rate Density [Counts $\mathrm{s}^{-1} \mathrm{ KeV}^{-1}$]')
plt.legend()
plt.yscale('log')
plt.ylim(0.00002, 0.4)
plt.savefig('BackgroundSpectrum.pdf')
plt.show()
