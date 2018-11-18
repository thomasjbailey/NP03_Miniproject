#a script to plot the experimental data for 137-Cs
#Author: Thomas Bailey
#Date Created: 20181118

import pandas as pd
import numpy as np
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
import uncertainties as u

#energy calibration function
def Calibrate(data, a, b):
    return a * data + b


data, metadata = h5load("../Data/Experimental/137-Cs.h5")

data['Energy'] = data['BinNumber'].apply(Calibrate, args=tuple(metadata['Calibration']))

sim_data = read_simulated_data("../Data/Simulated/Compton_14mm.txt")
#convert from MeV to KeV
sim_data['Energy'] = sim_data['Energy']*1000
#remove peak from photons not detected
sim_data['Total_counts'].iloc[0] = 0

#rescale simlulated data to match experimental
#number calculated from number of counts in main peak
sim_data['Total_counts'] = sim_data['Total_counts']*0.924

# plt.plot(data['Energy'], data['Counts'], label='Energy')
# plt.legend()
# plt.show()

plt.hist(unbin(data), bins=200, range=(0, 700), alpha=0.5, label='Experimental')
plt.hist(unbin_sim(sim_data), bins=200, range=(0, 700), alpha=0.5, label='Simulated')
plt.ylabel("Counts")
plt.xlabel("Energy (KeV)")
plt.legend()
plt.savefig('Compton_Edge.pdf')
plt.show()


#plt.plot(sim_data['Energy'], sim_data['Total_count'])
#plt.show()
