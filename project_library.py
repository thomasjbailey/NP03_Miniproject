#collection of functions used throughout the data analysis

import numpy as np
import pandas as pd
import random
import scipy.stats
import uncertainties as u
import uncertainties.umath as umath

#functions to write and open data in the HDF5 format with metadata
#source https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146
def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(filename):
    with pd.HDFStore(filename) as store:
        data = store['mydata']
        metadata = store.get_storer('mydata').attrs.metadata
        return data, metadata


#function to estimate unbin data by assuming each bin has values uniformly distributed
def unbin(data):
    bin_size = data['Energy'].iloc[1] - data['Energy'].iloc[0]
    unbinned_data = []
    for row in range(2048):
        for i in range(int(data['Counts'].iloc[row])):
            unbinned_data.append(data['Energy'].iloc[row]+bin_size*(random.random()-0.5))
    return unbinned_data

#function to estimate unbin simulated data
def unbin_sim(data):
    bin_size = data['Energy'].iloc[1] - data['Energy'].iloc[0]
    unbinned_data = []
    for row in range(data.shape[0]):
        for i in range(int(data['Total_counts'].iloc[row])):
            unbinned_data.append(data['Energy'].iloc[row]+bin_size*(random.random()-0.5))
    return unbinned_data

#function to read simulated data
def read_simulated_data(file):
    in_file = open(file)
    in_data = []
    for line in in_file:
    	line = line.rstrip('\n')
    	line = line.split(' ')
    	line = [float(i) for i in line]
    	in_data.append(line)
    in_data = np.asarray(in_data)
    return pd.DataFrame(data={'Energy': in_data[:,0], 'Input_Photons': in_data[:,1], 'Number_Full_energy_deposited': in_data[:, 2], 'Total_counts': in_data[:, 3]})

#gaussian function
def gaussian(x, mu, sig):
    return (1/np.sqrt(2*np.pi*(sig**2))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#function to fit to peaks
#gaussian with constant background
def peak_function(x, center, width, peak_count, background):
    return background + peak_count*gaussian(x, center, width)

#gaussian with constant background - probability density for likelihood fit
def peak_function2(x, center, width, signal, fit_width):
    return (1-signal)/fit_width + signal*gaussian(x, center, width)


#functions for efficiency calculations

def fit_value1(x, a, b, c, d, e, g, f):
	return a*umath.exp(b*x) + c*umath.exp(-(x**2)/d) + e*umath.exp(-(x*g))

def fit_value2(th, a, b, c, d, e, g, f):
	return a*umath.exp(b*th) + c*umath.exp(-(th**2)/d) + e*umath.exp(-(th**2)/g)

def fit_value3(th, a, b, c, d, e, g, f):
	return a*umath.exp(-(th**2)/b) + c*umath.exp(-(th**2)/d) + e*umath.exp(-(th**2)/g)

def fit_value4(th, a, b, c, d, f):
	return a*umath.exp(-th*b) + c*umath.exp(-(th**2)/d)

#calculate efficiency value using 3 different models and average
def Efficiency(Energy, coeffs):
    return (fit_value1(Energy, *coeffs[0]) + fit_value3(Energy, *coeffs[1]) + fit_value4(Energy, *coeffs[2]))/3 * coeffs[3]
