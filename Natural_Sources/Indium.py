#a scipt to compare indium spectra with Background
#Author: Thomas Bailey
#date created: 20181127

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
data_file = open("Background_data.txt", 'rb')
print('Loading Background Data from file')
Background_data = pickle.load(data_file)
Back_Counts = len(Background_data)
Back_Counts = u.ufloat(Back_Counts, np.sqrt(Back_Counts))
Back_Rate = Back_Counts / (164277+242316)
print(Back_Rate)

# back_data, back_metadata = h5load("../Data/Experimental/Short_Background.h5")
# Back_Counts = back_data['Counts'].sum()
# Back_Counts = u.ufloat(Back_Counts, np.sqrt(Back_Counts))
# Back_Rate = Back_Counts / back_metadata['RealTime']
# print(Back_Rate)
#
# back_data, back_metadata = h5load("../Data/Experimental/Background2.h5")
# Back_Counts = back_data['Counts'].sum()
# Back_Counts = u.ufloat(Back_Counts, np.sqrt(Back_Counts))
# Back_Rate = Back_Counts / back_metadata['RealTime']
# print(Back_Rate)
#
# back_data, back_metadata = h5load("../Data/Experimental/Background3.h5")
# Back_Counts = back_data['Counts'].sum()
# Back_Counts = u.ufloat(Back_Counts, np.sqrt(Back_Counts))
# Back_Rate = Back_Counts / back_metadata['RealTime']
# print(Back_Rate)

#no moderator
data_0, metadata_0 = h5load("../Data/Experimental/Indium_NoModerator.h5")
In_Counts_0 = data_0['Counts'].sum()
In_Counts_0 = u.ufloat(In_Counts_0, np.sqrt(In_Counts_0))
In_Rate_0 = In_Counts_0 / metadata_0['RealTime']
Excess_0 = In_Rate_0 - Back_Rate
Sigma_0 = Excess_0.n / Excess_0.s
P_0 = 1 - scipy.stats.norm.cdf(Sigma_0, 0, 1)
print(P_0)
print(Excess_0)

#1 moderator
data_1, metadata_1 = h5load("../Data/Experimental/Indium_1Moderator.h5")
In_Counts_1 = data_1['Counts'].sum()
In_Counts_1 = u.ufloat(In_Counts_1, np.sqrt(In_Counts_1))
In_Rate_1 = In_Counts_1 / metadata_1['RealTime']
Excess_1 = In_Rate_1 - Back_Rate
Sigma_1 = Excess_1.n / Excess_1.s
P_1 = 1 - scipy.stats.norm.cdf(Sigma_1, 0, 1)
print(P_1)
print(Excess_1)

#2 moderator
data_2, metadata_2 = h5load("../Data/Experimental/2moderatorIndium90min.h5")
In_Counts_2 = data_2['Counts'].sum()
In_Counts_2 = u.ufloat(In_Counts_2, np.sqrt(In_Counts_2))
In_Rate_2 = In_Counts_2 / metadata_2['RealTime']
Excess_2 = In_Rate_2 - Back_Rate
Sigma_2 = Excess_2.n / Excess_2.s
P_2 = 1 - scipy.stats.norm.cdf(Sigma_2, 0, 1)
print(P_2)
print(Excess_2)

#3 moderator
data_3, metadata_3 = h5load("../Data/Experimental/3moderatorIndium70min.h5")
In_Counts_3 = data_3['Counts'].sum()
In_Counts_3 = u.ufloat(In_Counts_3, np.sqrt(In_Counts_3))
In_Rate_3 = In_Counts_3 / metadata_3['RealTime']
Excess_3 = In_Rate_3 - Back_Rate
Sigma_3 = Excess_3.n / Excess_3.s
P_3 = 1 - scipy.stats.norm.cdf(Sigma_3, 0, 1)
print(P_3)
print(Excess_3)

#4 moderator
data_4, metadata_4 = h5load("../Data/Experimental/Indium_4Moderator.h5")
In_Counts_4 = data_4['Counts'].sum()
In_Counts_4 = u.ufloat(In_Counts_4, np.sqrt(In_Counts_4))
In_Rate_4 = In_Counts_4 / metadata_4['RealTime']
Excess_4 = In_Rate_4 - Back_Rate
Sigma_4 = Excess_4.n / Excess_4.s
P_4 = 1 - scipy.stats.norm.cdf(Sigma_4, 0, 1)
print(P_4)
print(Excess_4)

print(In_Rate_0)
print(In_Rate_1)
print(In_Rate_2)
print(In_Rate_3)
print(In_Rate_4)


#find decay constant
measure_times = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
counts = []
count_uncertainties = []
times = []
for i in range(7):
    data, metadata = h5load("../Data/Experimental/3moderatorIndium"+measure_times[i]+"min.h5")
    counts.append((u.ufloat(data['Counts'].sum(), np.sqrt(data['Counts'].sum())) - (Back_Rate * metadata['RealTime'])).n)
    count_uncertainties.append((u.ufloat(data['Counts'].sum(), np.sqrt(data['Counts'].sum())) - (Back_Rate * metadata['RealTime'])).s)
    times.append(metadata['RealTime'])


for i in range(9):
    data, metadata = h5load("../Data/Experimental/2moderatorIndium"+measure_times[i]+"min.h5")
    counts.append((u.ufloat(data['Counts'].sum(), np.sqrt(data['Counts'].sum())) - (Back_Rate * metadata['RealTime'])).n)
    count_uncertainties.append((u.ufloat(data['Counts'].sum(), np.sqrt(data['Counts'].sum())) - (Back_Rate * metadata['RealTime'])).s)
    times.append(metadata['RealTime'])

counts += [(In_Counts_0 - (Back_Rate * metadata_0['RealTime'])).n]
count_uncertainties += [(In_Counts_0 - (Back_Rate * metadata_0['RealTime'])).s]
times += [metadata_0['RealTime']]

counts += [(In_Counts_1 - (Back_Rate * metadata_1['RealTime'])).n]
count_uncertainties += [(In_Counts_1 - (Back_Rate * metadata_1['RealTime'])).s]
times += [metadata_1['RealTime']]

counts += [(In_Counts_4 - (Back_Rate * metadata_4['RealTime'])).n]
count_uncertainties += [(In_Counts_4 - (Back_Rate * metadata_4['RealTime'])).s]
times += [metadata_4['RealTime']]

plt.errorbar(times, counts, count_uncertainties, linestyle='none', marker='x')
plt.show()

#plt.plot(data_0['BinNumber'], ((data_0['Counts'] / metadata_0['RealTime']) + (data_1['Counts'] / metadata_1['RealTime']) + (data_2['Counts'] / metadata_2['RealTime']) + (data_3['Counts'] / metadata_3['RealTime']) + (data_4['Counts'] / metadata_4['RealTime']))/5, alpha = 0.5, label = 'indium')

data_0['Energy'] = data_0['BinNumber'].apply(calibrate_array, args=tuple(metadata_0['Calibration']))
data_1['Energy'] = data_1['BinNumber'].apply(calibrate_array, args=tuple(metadata_1['Calibration']))
data_2['Energy'] = data_2['BinNumber'].apply(calibrate_array, args=tuple(metadata_2['Calibration']))
data_3['Energy'] = data_3['BinNumber'].apply(calibrate_array, args=tuple(metadata_3['Calibration']))
data_4['Energy'] = data_4['BinNumber'].apply(calibrate_array, args=tuple(metadata_4['Calibration']))

indium_data = unbin(data_0)
indium_data += unbin(data_1)
indium_data += unbin(data_2)
indium_data += unbin(data_3)
indium_data += unbin(data_4)

indium_time = metadata_0['RealTime'] + metadata_1['RealTime'] + metadata_2['RealTime'] + metadata_3['RealTime'] + metadata_4['RealTime']

binned_indium, edges = np.histogram(indium_data, 200)
indium_uncertainty = np.sqrt(binned_indium)
binned_indium = binned_indium / indium_time
indium_uncertainty = indium_uncertainty / indium_time
centers = [(edges[i]+edges[i+1])/2 for i in range(len(binned_indium))]
w = (centers[1]-centers[0])
binned_indium = binned_indium / w
indium_uncertainty = indium_uncertainty / w
plt.bar(centers, binned_indium, width = w, alpha =0.5, align='center', label='Indium', color = 'C0', yerr=indium_uncertainty, error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C0', 'capsize': 2})


binned_background, edges2 = np.histogram(Background_data, 200)
background_uncertainty = np.sqrt(binned_background)
binned_background = binned_background / (164277+242316)
background_uncertainty = background_uncertainty / (164277+242316)
centers2 = [(edges2[i]+edges2[i+1])/2 for i in range(len(binned_background))]
w2 = (centers2[1]-centers2[0])
binned_background = binned_background / w2
background_uncertainty = background_uncertainty / w2
plt.bar(centers2, binned_background, width = w2, alpha =0.5, align='center', label='Background', yerr=background_uncertainty, color='C1', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C1', 'capsize': 2})

plt.xlabel('Energy [KeV]')
plt.ylabel(r'Count Rate Density [Counts $\mathrm{s}^{-1} \mathrm{ KeV}^{-1}$]')
plt.legend()
plt.savefig('IndiumSpectrum.pdf')
plt.show()
