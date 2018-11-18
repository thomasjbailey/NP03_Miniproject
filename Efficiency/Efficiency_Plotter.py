#a script to calculate the efficiency of the detector from experimental data
#Author: Thomas Bailey
#Creation Date: 20181118

import numpy as np
import pandas as pd
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *
import matplotlib.pyplot as plt
import uncertainties as u
import pickle

with open('Thorium_Intensity.txt', 'rb') as fp:
    counts_in_peak_Th = pickle.load(fp)

Th_peaks = [99.5, 105.3, 115.2, 129.1, 154.2, 209.4, 238.6, 270.3, 278, 300.1, 328, 338.4, 409.4, 463, 583.1, 727, 794.8, 860.4, 911.1]

known_probabilities_Th = np.array([u.ufloat(1.30, 0.07),
                        u.ufloat(4.8, 0.3),
                        u.ufloat(0.71, 0.05),
                        u.ufloat(2.7, 0.2),
                        u.ufloat(0.79, 0.05),
                        u.ufloat(3.9, 0.2),
                        u.ufloat(43.0, 2.0),
                        u.ufloat(3.3, 0.2),
                        u.ufloat(2.4, 0.1),
                        u.ufloat(2.9, 0.2),
                        u.ufloat(2.9, 0.2),
                        u.ufloat(10.7, 0.5),
                        u.ufloat(1.8, 0.1),
                        u.ufloat(4.2, 0.2),
                        u.ufloat(30.0, 1.4),
                        u.ufloat(6.9, 0.4),
                        u.ufloat(4.2, 0.2),
                        u.ufloat(4.2, 0.2),
                        u.ufloat(25.4, 1.3)])

Th_efficiency = np.divide(np.array(counts_in_peak_Th),known_probabilities_Th)

Th_efficiency_n = []
Th_efficiency_s = []
for i in range(len(Th_efficiency)):
	Th_efficiency_n.append(Th_efficiency[i].n)
	Th_efficiency_s.append(Th_efficiency[i].s)

plt.errorbar(Th_peaks, Th_efficiency_n, Th_efficiency_s, linestyle='', fmt='x', label='Th')


with open('Europium_Intensity.txt', 'rb') as fp:
    counts_in_peak_Eu = pickle.load(fp)

Eu_peaks = [121.78, 244.7, 344.28, 411.12, 443.96, 778.9, 867.37, 964.08, 1005.3, 1085.9, 1112.1, 1299.1, 1408]

known_probabilities_Eu = np.array([
                        u.ufloat(28.58, 0.09),
                        u.ufloat(7.580, 0.030),
                        u.ufloat(26.5, 0.6),
                        u.ufloat(2.234, 0.025),
                        u.ufloat(3.148, 0.020),
                        u.ufloat(12.94, 0.15),
                        u.ufloat(4.245, 0.021),
                        u.ufloat(14.60, 0.04),
                        u.ufloat(0.646, 0.005),
                        (u.ufloat(10.21, 0.04)+u.ufloat(1.727,0.020)),
                        u.ufloat(13.64, 0.04),
                        u.ufloat(1.623, 0.020),
                        u.ufloat(21.00, 0.05)
                        ])

Eu_efficiency = np.divide(np.array(counts_in_peak_Eu),known_probabilities_Eu)

Eu_efficiency_n = []
Eu_efficiency_s = []
for i in range(len(Eu_efficiency)):
	Eu_efficiency_n.append(Eu_efficiency[i].n)
	Eu_efficiency_s.append(Eu_efficiency[i].s)

plt.errorbar(Eu_peaks, Eu_efficiency_n, Eu_efficiency_s, linestyle='', fmt='x', label='Eu')
plt.legend()
plt.show()
