#a scipt to compare brazil nut spectra with Background
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

#functions to calibrate the brazil nut spectrum

#function used as the fitted calibration
#in this order so that bin peak can be on the y-axis during fit
def calibration_fit_function(energy, a, b):
    return (energy - b) / a

#energy calibration functions
def calibrate_array(data, a, b):
    return a * data + b

# fit a peak to a region of data with width fit_width around the peak bin
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

    # plt.plot(df['BinNumber'], np.log(df['Counts'].astype(int)+1), 'x')

    known_energies = [74.97, 238.63, 295.22, 351.93, 511, 583.19, 609.31, 1460.82]
    #perform a fit for each peak to find the central value
    for peak in known_energies:
        #fit peak around bin number expected from meta data calibration
        fit_values, fit_errors = peak_fit(df, (peak-metadata['Calibration'][1])/metadata['Calibration'][0], 30)
        peak_bins.append(fit_values[0])
        peak_bin_uncertainties.append(np.sqrt(fit_errors[0]))
    #     plt.plot(np.linspace(peak-25, peak+25), peak_function(np.linspace(peak-25, peak+25), *fit_values))
    # plt.show()

    lin_fit, pcov = curve_fit(calibration_fit_function, known_energies, peak_bins, sigma=peak_bin_uncertainties)
    df['Energy'] = df['BinNumber'].apply(calibrate_array, args=tuple(lin_fit))
    return df



#functions to perform unbinned fits to the data datasets

#function to perform a likelihood fit to the data
def unbinned_exp_LLH(data, starting_param, param_limit):
    # Create an unbinned likelihood object with function and data.
    unbin = probfit.UnbinnedLH(peak_function2, data)

    # Minimizes the unbinned likelihood for the given function
    m = iminuit.Minuit(unbin,
                       **starting_param,
                       limit_center = param_limit['limit_center'],
                       limit_width = param_limit['limit_width'],
                       limit_signal = param_limit['limit_signal'],
                       fix_fit_width = True,
                       pedantic=False,
                       print_level=1)
    m.migrad()
    params = m.values.values() # Get out fit values
    errs   = m.errors.values()
    return params, errs

#function to perform a fit to the specified peak and return its size
def peak_size_finder(peak, data, fit_width):
    print('Fitting to Peak at Energy: {} KeV'.format(peak))
    data = list(filter(lambda x: x >= peak - fit_width/2, data))
    data = list(filter(lambda x: x <= peak + fit_width/2, data))
    number_counts = len(data)
    data = np.array(data)

    starting_param = {'center':peak, 'width':1, 'signal':0.5, 'fit_width':fit_width}
    param_limit = {'limit_center':(peak-3, peak+3), 'limit_width':(0.01, 5), 'limit_signal':(0.01, 0.99)}

    params, errs = unbinned_exp_LLH(data, starting_param, param_limit)

    # Plot
    x_pts = np.linspace(peak-20, peak+20, 100)
    plt.plot(x_pts, number_counts*peak_function2(x_pts, *params), color = "black")
    plt.hist(data, color = "lightgrey", bins = fit_width)
    plt.ylabel('Counts')
    plt.xlabel('Energy [KeV]')
    plt.savefig('Intensity_plots/'+str(peak)+'.pdf')
    plt.close()
    counts_in_peak = u.ufloat(params[2]*number_counts,errs[2]*number_counts)
    print(counts_in_peak)
    return counts_in_peak



data, metadata = h5load("../Data/Experimental/Brazil_nuts.h5")
print(metadata)
#perform an energy calibration to the brazil nut spectrum
data = Calibrate(data, metadata)
unbinned_brazil = unbin(data)

fit_width = 8
Brazil_peaks = [74.97, 238.63, 295.22, 351.93, 511, 583.19, 609.31, 1460.82]

# print('Finding Peak Sizes')
# counts_in_peak_Brazil = Parallel(n_jobs=-1)(delayed(peak_size_finder)(peak, unbinned_brazil, fit_width) for peak in Brazil_peaks)
# print(counts_in_peak_Brazil)
#
# #save peak values
# with open('Brazil_Intensity.txt', 'wb') as fp:
#     pickle.dump(counts_in_peak_Brazil, fp)

#open brazil peak data
with open('Brazil_Intensity.txt', 'rb') as fp:
    counts_in_peak_Brazil = pickle.load(fp)

# #get values of counts in each peak for background
data_file = open("Background_data.txt", 'rb')
print('Loading Background Data from file')
Background_data = pickle.load(data_file)
# fit_width = 8
# Brazil_peaks = [74.97, 238.63, 295.22, 351.93, 511, 583.19, 609.31, 1460.82]
#
# print('Finding Peak Sizes')
# counts_in_peak_Background = Parallel(n_jobs=-1)(delayed(peak_size_finder)(peak, Background_data, fit_width) for peak in Brazil_peaks)
# print(counts_in_peak_Background)
#
# #save peak values
# with open('Background_Intensity.txt', 'wb') as fp:
#     pickle.dump(counts_in_peak_Background, fp)

#open background peak data
with open('Background_Intensity.txt', 'rb') as fp:
    counts_in_peak_Background = pickle.load(fp)

#detector run times for the two data sets
brazil_time = u.ufloat(63203, 1)
background_time = u.ufloat(164277+242316,1.4)

#efficiency function paramaters calculated in Efficiency_Plotter scipt
with open('../Efficiency/EfficiencyParamaters.txt', 'rb') as fp:
    coeffs = pickle.load(fp)

#dictionary to store activity of each decay
activities = {}
#dictionary to store p-value of signal above background
p_values = {}

for i in range(len(Brazil_peaks)):
    Brazil_signal = counts_in_peak_Brazil[i]
    Brazil_signal = u.ufloat(Brazil_signal.n, np.sqrt(Brazil_signal.s**2 + Brazil_signal.n))
    Brazil_signal = Brazil_signal/brazil_time

    Background_signal = counts_in_peak_Background[i]
    Background_signal = u.ufloat(Background_signal.n, np.sqrt(Background_signal.s**2 + Background_signal.n))
    Background_signal = Background_signal/background_time

    corrected_count = (Brazil_signal-Background_signal)
    sigma = corrected_count.n/corrected_count.s
    print("Sigma at {} KeV: {}".format(Brazil_peaks[i], sigma))
    #calculate p-value assuming the uncertainties follow a normal distribution
    p_value = 1-scipy.stats.norm.cdf(sigma, 0 ,1)
    p_values[Brazil_peaks[i]] = p_value

    activity = corrected_count / Efficiency(Brazil_peaks[i], coeffs)
    activities[Brazil_peaks[i]] = activity


#use Fisher's method to combine p-values for each decay chain

#Thorium Decays
X_2 = -2*(np.log(p_values[238.63])+np.log(p_values[583.19]))
DoF = 4
Th_p_value = 1 - scipy.stats.chi2.cdf(X_2, DoF)
print(Th_p_value)

#Radium Decays
X_2 = -2*(np.log(p_values[295.22])+np.log(p_values[351.93])+np.log(p_values[609.31]))
DoF = 6
Ra_p_value = 1 - scipy.stats.chi2.cdf(X_2, DoF)
print(Ra_p_value)


#calculate % of each element given its activity
mass_of_nuts = u.ufloat(150, 150*0.045)

#for Radium 228 decays (from Thorium)
#assume no thorium in the nuts
Th_activity = 0.5*(activities[238.63]/u.ufloat(0.430,0.020) + activities[583.19]/u.ufloat(0.30,0.014))
print(Th_activity)
#activity of one mole of thorium
Th_Ac_mole = c.N_A * (np.log(2) / (u.ufloat(1.40,0.01)*(10**10)*365*24*60*60))
#activity of one mole of Ra-228
Ra228_Ac_mole = c.N_A * (np.log(2) / (u.ufloat(5.75,0.03)*365*24*60*60))
#in 'steady' state all radio nucleides will have the same activity
#the number of moles of Ra-228 for every mole of Th-232 in steady state
mole_ratio = Th_Ac_mole / Ra228_Ac_mole
print(mole_ratio)
#number of moles of Thorium = activity / activity of one mole
Th_mole = Th_activity / Th_Ac_mole
Th_RFM = u.ufloat(232.0380495, 0.0000022)
Th_mass = Th_RFM * Th_mole
Ra228_mole = mole_ratio * Th_mole
Ra228_RFM = u.ufloat(228.031, 0.0005)
Ra228_mass = Ra228_mole * Ra228_RFM
print("Mass of Thorium: {}g".format(Th_mass))
print("Mass abundance: {}%".format(100*Th_mass/mass_of_nuts))
print("Mass of Ra-228: {}g".format(Ra228_mass))
print("Mass abundance of Ra-228: {}%".format(100*Ra228_mass/mass_of_nuts))

Ra226_activity = (activities[295.22]/u.ufloat(0.18414,0.00036) + activities[351.93]/u.ufloat(0.3560,0.0017) + activities[609.31]/u.ufloat(0.4549,0.0019))/3
print(Ra226_activity)
#activity of one mole of Ra-226
Ra226_Ac_mole = c.N_A * (np.log(2) / (u.ufloat(1600,7)*365*24*60*60))
#number of moles of Radium = activity / activity of one mole
Ra226_mole = Ra226_activity / Ra226_Ac_mole
Ra226_RFM = u.ufloat(226.025, 0.0005)
Ra226_mass = Ra226_RFM * Ra226_mole
print("Mass of Ra-226: {}g".format(Ra226_mass))
print("Mass abundance: {}%".format(100*Ra226_mass/mass_of_nuts))

K40_activity = activities[1460.82]/u.ufloat(0.1066,0.0017)
print(K40_activity)
#activity of one mole of Ra-226
K40_Ac_mole = c.N_A * (np.log(2) / (u.ufloat(1.248,0.003)*(10**9)*365*24*60*60))
#number of moles of Radium = activity / activity of one mole
K40_mole = K40_activity / K40_Ac_mole
K40_RFM = u.ufloat(39.964, 0.0005)
K40_mass = K40_RFM * K40_mole
print("Mass of K-40: {}g".format(K40_mass))
print("Mass abundance: {}%".format(100*K40_mass/mass_of_nuts))



#plot Brazil nut and background spectra
binned_brazil, edges = np.histogram(unbinned_brazil, 200)
brazil_uncertainty = np.sqrt(binned_brazil)
binned_brazil = binned_brazil / brazil_time.n
brazil_uncertainty = brazil_uncertainty / brazil_time.n
centers = [(edges[i]+edges[i+1])/2 for i in range(len(binned_brazil))]
w = (centers[1]-centers[0])
binned_brazil = binned_brazil / w
brazil_uncertainty = brazil_uncertainty / w
plt.bar(centers, binned_brazil, width = w, alpha =0.5, align='center', label='Brazil Nuts', color = 'C0', yerr=brazil_uncertainty, error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C0', 'capsize': 2})

binned_background, edges2 = np.histogram(Background_data, 200)
background_uncertainty = np.sqrt(binned_background)
binned_background = binned_background / background_time.n
background_uncertainty = background_uncertainty / background_time.n
centers2 = [(edges2[i]+edges2[i+1])/2 for i in range(len(binned_background))]
w2 = (centers2[1]-centers2[0])
binned_background = binned_background / w2
background_uncertainty = background_uncertainty / w2
plt.bar(centers2, binned_background, width = w2, alpha =0.5, align='center', label='Background', yerr=background_uncertainty, color='C1', error_kw={'elinewidth':1, 'errorevery': 10, 'ecolor': 'C1', 'capsize': 2})

plt.legend()
plt.ylabel('Count Rate Density [Counts/second/KeV]')
plt.xlabel('Energy [KeV]')
plt.yscale('log')
plt.ylim(0.00001, 0.008)
plt.savefig('BrazilNutSpectrum.pdf')
plt.show()
