#collection of functions used throughout the data analysis

import numpy as np
import pandas as pd
import random
import scipy.stats

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
    #return decay*(1-signal)*np.exp(-decay*x) + signal*gaussian(x, center, width)
    return (1-signal)/fit_width + signal*gaussian(x, center, width)
