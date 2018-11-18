#a script to calculate the efficiency of the detector at different known_energies
#Author: Thomas Bailey
#Date Created: 20181118

import numpy as np
import pandas as pd
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
import uncertainties as u
import pickle
import iminuit, probfit
import scipy.stats
from joblib import Parallel, delayed

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
    plt.show()
    return u.ufloat(params[2]*number_counts,errs[2]*number_counts)

#get values of counts in each peak for Thorium
data_file = open("Thorium_data.txt", 'rb')
print('Loading Thorium Data from file')
Thorium_data = pickle.load(data_file)
fit_width = 10
Th_peaks = [99.5, 105.3, 115.2, 129.1, 154.2, 209.4, 238.6, 270.3, 278, 300.1, 328, 338.4, 409.4, 463, 583.1, 727, 794.8, 860.4, 911.1]

#plt.hist(Thorium_data, 2000, log=True)
#plt.show()

print('Finding Peak Sizes')
#counts_in_peak_Th = Parallel(n_jobs=-1)(delayed(peak_size_finder)(peak, Thorium_data, fit_width) for peak in Th_peaks)
#print(counts_in_peak_Th)

#save peak values
#with open('Thorium_Intensity.txt', 'wb') as fp:
#    pickle.dump(counts_in_peak_Th, fp)


#get values of counts in each peak for Europium
data_file = open("Europium_data.txt", 'rb')
print('Loading Europium Data from file')
Europium_data = pickle.load(data_file)
fit_width = 15
Eu_peaks = [121.78, 244.7, 344.28, 411.12, 443.96, 778.9, 867.37, 964.08, 1005.3, 1085.9, 1112.1, 1299.1, 1408]

# plt.hist(Europium_data, 2000, log=True)
# plt.show()

print('Finding Peak Sizes')
counts_in_peak_Eu = Parallel(n_jobs=-1)(delayed(peak_size_finder)(peak, Europium_data, fit_width) for peak in Eu_peaks)

print(counts_in_peak_Eu)

#save peak values
with open('Europium_Intensity.txt', 'wb') as fp:
    pickle.dump(counts_in_peak_Eu, fp)
