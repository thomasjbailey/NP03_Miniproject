#a script to calculate the efficiency of the detector at different known_energies
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
import iminuit, probfit
import scipy.stats

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

data_file = open("Europium_data.txt", 'rb')
Europium_data = pickle.load(data_file)
fit_width = 40
peak = 1299.1
data = list(filter(lambda x: x >= peak - fit_width/2, Europium_data))
data = list(filter(lambda x: x <= peak + fit_width/2, data))
data = np.array(data)
number_counts = data.shape[0]
print(number_counts)

# order of stating paramaters values
#center_init, width_init, peak_count_init, background_init
starting_param = {'center':peak, 'width':2, 'signal':0.8, 'fit_width':fit_width}
param_limit = {'limit_center':(-1290, 1310), 'limit_width':(0.01, 5), 'limit_signal':(0.1, 0.9)}

params, errs = unbinned_exp_LLH(data, starting_param, param_limit)

print(params)

center, width, signal, fit_width = params
print(signal*number_counts)
print(errs[2]*number_counts)

# Plot
x_pts = np.linspace(1280, 1320, 100)
plt.plot(x_pts, number_counts*peak_function2(x_pts, *params), color = "black")
plt.hist(data, color = "lightgrey", bins = 40)
#plt.xlim(0, 3)
#plt.legend()
plt.show()
