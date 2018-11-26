#a script to calculate the number of counts in the 40-K peak
#Author: Thomas Bailey
#Date Created: 20181126

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

#get values of counts in each peak for Thorium
data_file = open("Potassium_data.txt", 'rb')
print('Loading Potassium Data from file')
Potassium_data = pickle.load(data_file)
fit_width = 8
K_peak = 1460.8

plt.hist(Potassium_data, 2000, log=True)
plt.show()

print('Finding Peak Sizes')
counts_in_peak_K = peak_size_finder(K_peak, Potassium_data, fit_width)
print(counts_in_peak_K)

#save peak values
with open('Potassium_Intensity.txt', 'wb') as fp:
   pickle.dump(counts_in_peak_K, fp)
