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

sim_data = read_simulated_data("../Data/Simulated/Compton_10mm.txt")
#convert from MeV to KeV
sim_data['Energy'] = sim_data['Energy']*1000
#remove peak from photons not detected
sim_data['Total_counts'].iloc[0] = 0

#rescale simlulated data to match experimental
#number calculated from number of counts in main peak
sim_peak = sim_data['Total_counts'].loc[sim_data['Energy']>642].loc[sim_data['Energy']<656].sum()
exp_peak = data['Counts'].loc[data['Energy']>657].loc[data['Energy']<665].sum()

sim_data['Total_counts'] = sim_data['Total_counts']*(exp_peak/sim_peak)

# plt.plot(data['Energy'], data['Counts'], label='Energy')
# plt.legend()
# plt.show()

plt.hist(unbin(data), bins=200, range=(0, 700), alpha=0.5, label='Experimental')
plt.hist(unbin_sim(sim_data), bins=200, range=(0, 700), alpha=0.5, label='Simulated')
ax = plt.gca()
ax.annotate('Reverse Compton Edge', xy=(187, 3600), xytext=(190, 7000),
            arrowprops=dict(arrowstyle="->"),
            )
ax.annotate('Compton Edge', xy=(467, 2400), xytext=(450, 5000),
            arrowprops=dict(arrowstyle="->"))

plt.ylabel("Counts")
plt.xlabel("Energy [KeV]")
plt.legend()
plt.savefig('Compton_Edge.pdf')
plt.show()


#plt.plot(sim_data['Energy'], sim_data['Total_count'])
#plt.show()
