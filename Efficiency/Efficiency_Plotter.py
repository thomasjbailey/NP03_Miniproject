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
from scipy.optimize import curve_fit
import glob
import scipy.interpolate as inter
import scipy.constants as c
import uncertainties.umath as umath

#import the fitted peak counts from file
#a list with fitted number of counts in each peak
with open('Thorium_Intensity.txt', 'rb') as fp:
    counts_in_peak_Th = pickle.load(fp)

#known energies of each thorium gamma peak
Th_peaks = [99.5, 105.3, 115.2, 129.1, 154.2, 209.4, 238.6, 270.3, 278, 300.1, 328, 338.4, 409.4, 463, 583.1, 727, 794.8, 860.4, 911.1]

#known probability of each thorium gamma peak
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

#calculate efficiency
Th_efficiency = np.divide(np.array(counts_in_peak_Th),known_probabilities_Th)

#convert from numpy arrays with ufloats to lists with normal floats
Th_efficiency_n = []
Th_efficiency_s = []
for i in range(len(Th_efficiency)):
	Th_efficiency_n.append(Th_efficiency[i].n)
	Th_efficiency_s.append(Th_efficiency[i].s)


#open peak intensity data from fits for Europium
with open('Europium_Intensity.txt', 'rb') as fp:
    counts_in_peak_Eu = pickle.load(fp)

#known energies and probabilities of each peak
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

#convert from numpy array of ufloats to list of floats
Eu_efficiency_n = []
Eu_efficiency_s = []
for i in range(len(Eu_efficiency)):
	Eu_efficiency_n.append(Eu_efficiency[i].n)
	Eu_efficiency_s.append(Eu_efficiency[i].s)


#fit a joint fit to the two datasets
x = np.hstack((Th_peaks, Eu_peaks))
y = np.hstack((Th_efficiency_n, Eu_efficiency_n))
y_err = np.hstack((Th_efficiency_s, Eu_efficiency_s))

#function to fit the two data sets with a multiplicative constant between their efficiencies
#use a model of two exponentials and a gaussian to fit the efficiency data
def fit(x, a, b, c, d, e, g, f):
	th = x[0:19]
	eu = x[19:]
	th = a*np.exp(b*th) + c*np.exp(-(th**2)/d) + e*np.exp(-(th*g))
	eu = f*(a*np.exp(b*eu) + c*np.exp(-(eu**2)/d) + e*np.exp(-(eu*g)))
	return np.hstack([th, eu])

#function to plot the final fit
def fit_plotter(x, a, b, c, d, e, g, f):
	return a*np.exp(b*x) + c*np.exp(-(x**2)/d) + e*np.exp(-(x*g))

#perform a fit to the data
popt, pcov = curve_fit(fit, x, y, p0=[70, -0.001, 400, 35000, 200, 0.01, 16], sigma=y_err)
print(popt)



#read the simulated efficiency for an active thickness of 10mm
#a panda dataframe with headings 'Energy', 'Input_Photons', 'Number_Full_energy_deposited', 'Total_counts'
sim_data = read_simulated_data("../Data/Simulated/AllEnergies_10mm.txt")

#remove entries at energies where there are no inputted photons and the entry at 0.22 MeV that causes a bad spline fit
#energies converted from MeV to KeV
x_sim = (sim_data['Energy'].loc[sim_data['Energy']>=0.03].loc[sim_data['Energy']!=0.22].values)*1000
#y_sim = efficiency (total that deposited full energy/total inputted at that energy)
y_sim = np.divide(sim_data['Number_Full_energy_deposited'].loc[sim_data['Energy']>=0.03].loc[sim_data['Energy']!=0.22].values, sim_data['Input_Photons'].loc[sim_data['Energy']>=0.03].loc[sim_data['Energy']!=0.22].values)

#create a weight for the spline to ensure that the leveling off at 100% is fitted without overfitting elsewhere
w = np.arange(1, 297)
w = 1/w

#fit and plot a spline for the simulated data
s1 = inter.UnivariateSpline (x_sim, y_sim, s=0.00001, w=w)


#open peak intensity data for K-40
with open('Potassium_Intensity.txt', 'rb') as fp:
    K_peak = pickle.load(fp)
K_time = u.ufloat(6551, 1.4) #duration of the KCl run duration

#calculate the expected activity of the KCl source
Mass = u.ufloat(353.93, 0.005)
Molar_Mass = u.ufloat(74.548, 0.0005)
K40_abundance = u.ufloat(0.000117, 0.000001)
half_life = u.ufloat(1.248, 0.003)*(10**9)*365*24*60*60
gamma_decay_mode = u.ufloat(10.72, 0.11) * 0.01 #fraction of decays with a gamma emmission

moles_K40 = (Mass/Molar_Mass)*K40_abundance
number_K40 = c.N_A * moles_K40
print(number_K40)
decay_constant = np.log(2)/half_life
activity = decay_constant * number_K40
gamma_activity = activity * gamma_decay_mode
print("The expected activity for the low Na Salt is {}".format(gamma_activity))

#function to extract efficiency with uncertainties
def fit_value1(x, a, b, c, d, e, g, f):
	return a*umath.exp(b*x) + c*umath.exp(-(x**2)/d) + e*umath.exp(-(x*g))

#calculate expected signal from efficiency curve if all photons passed through detector
#efficeiency from fit to experimental data
Exp_eff_1460 = fit_value1(1460.82, *u.correlated_values(popt, pcov))
#expected count rate
count_rate_1 = gamma_activity * Exp_eff_1460
print(Exp_eff_1460)


#to check systematic errors in the empirical model used try some other fit functions

#function to fit the two data sets with a multiplicative constant between their efficiencies
#use a model of two gaussians and an expontential to fit the efficiency data
def fit2(x, a, b, c, d, e, g, f):
	th = x[0:19]
	eu = x[19:]
	th = a*np.exp(b*th) + c*np.exp(-(th**2)/d) + e*np.exp(-(th**2)/g)
	eu = f*(a*np.exp(b*eu) + c*np.exp(-(eu**2)/d) + e*np.exp(-(eu**2)/g))
	return np.hstack([th, eu])

#function to plot the final fit
def fit_value2(th, a, b, c, d, e, g, f):
	return a*umath.exp(b*th) + c*umath.exp(-(th**2)/d) + e*umath.exp(-(th**2)/g)

#perform a fit to the data
popt2, pcov2 = curve_fit(fit2, x, y, p0=[35, -0.001, 400, 25000, 58, 200000, 20], sigma=y_err)

#calculate expected signal from efficiency curve if all photons passed through detector
#efficeiency from fit to experimental data
Exp_eff_1460 = fit_value2(1460.82, *u.correlated_values(popt2, pcov2))
#expected count rate
count_rate_2 = gamma_activity * Exp_eff_1460
print(Exp_eff_1460)

#fit with 3 gaussians
def fit3(x, a, b, c, d, e, g, f):
	th = x[0:19]
	eu = x[19:]
	th = a*np.exp(-(th**2)/b) + c*np.exp(-(th**2)/d) + e*np.exp(-(th**2)/g)
	eu = f*(a*np.exp(-(eu**2)/b) + c*np.exp(-(eu**2)/d) + e*np.exp(-(eu**2)/g))
	return np.hstack([th, eu])

#function to plot the final fit
def fit_value3(th, a, b, c, d, e, g, f):
	return a*umath.exp(-(th**2)/b) + c*umath.exp(-(th**2)/d) + e*umath.exp(-(th**2)/g)

#perform a fit to the data
popt3, pcov3 = curve_fit(fit3, x, y, p0=[20, 30000000, 400, 25000, 58, 200000, 20], sigma=y_err)

#calculate expected signal from efficiency curve if all photons passed through detector
#efficeiency from fit to experimental data
Exp_eff_1460 = fit_value3(1460.82, *u.correlated_values(popt3, pcov3))
#expected count rate
count_rate_3 = gamma_activity * Exp_eff_1460
print(Exp_eff_1460)


#fit with 1 gaussian and 1 exponential
def fit4(x, a, b, c, d, f):
	th = x[0:19]
	eu = x[19:]
	th = a*np.exp(-th*b) + c*np.exp(-(th**2)/d)
	eu = f*(a*np.exp(-eu*b) + c*np.exp(-(eu**2)/d))
	return np.hstack([th, eu])

#function to plot the final fit
def fit_value4(th, a, b, c, d, f):
	return a*umath.exp(-th*b) + c*umath.exp(-(th**2)/d)

#perform a fit to the data
popt4, pcov4 = curve_fit(fit4, x, y, p0=[60, 0.001, 400, 35000, 20], sigma=y_err)

#calculate expected signal from efficiency curve if all photons passed through detector
#efficeiency from fit to experimental data
Exp_eff_1460 = fit_value4(1460.82, *u.correlated_values(popt4, pcov4))
#expected count rate
count_rate_4 = gamma_activity * Exp_eff_1460
print(Exp_eff_1460)


#using uncertainties from the fits and combining
#remove fit 2 from average as it has a strangely high uncertainty (although the fit looks good)
average_count = (count_rate_1 + count_rate_3 + count_rate_4)/3


#instead estimate uncertainty using standard deviation in fit measurements
#count_rate = np.array([count_rate_1.n, count_rate_2.n, count_rate_3.n, count_rate_4.n])
#average_count = u.ufloat(np.mean(count_rate), np.std(count_rate))


#experimental count rate
exp_count_rate = K_peak / K_time

#the overall detector efficiency
efficiency_coeff = exp_count_rate / average_count
print(efficiency_coeff)


#plot the efficiency data
plt.errorbar(Eu_peaks, np.array(Eu_efficiency_n)*efficiency_coeff.n/popt[-1], np.array(Eu_efficiency_s)*efficiency_coeff.n/popt[-1], linestyle='None', marker='x',markersize=7, elinewidth=1, color='blue', label='Europium Data')
plt.errorbar(Th_peaks, np.array(Th_efficiency_n)*efficiency_coeff.n, np.array(Th_efficiency_s)*efficiency_coeff.n, linestyle='None', marker='o', markerfacecolor='none', elinewidth=1, color='green', label='Thorium Data')
#plt.errorbar(data_th['Energy'].values[0:2], nominal_efficiency_th[0:2]/(fit_plotter(0, *popt)), uncertainty_efficiency_th[0:2]/(fit_plotter(0, *popt)), linestyle='None', marker='x', elinewidth=1, color='green', alpha=0.5)

#plt the fit line
plt.plot(np.linspace(0,1500), fit_plotter(np.linspace(0, 1500), *popt)*efficiency_coeff.n, color='black', label='Experimental Fit')


plt.plot (x_sim, s1(x_sim)*efficiency_coeff.n*fit_plotter(0, *popt), '--r', label='Simulated Efficiency')
#plt.plot(sim_data['Energy']*1000, np.divide(sim_data['Number_Full_energy_deposited'].values, sim_data['Input_Photons'].values), 'x', label='Simulated', alpha=0.2)

plt.xlim(0,1500)

plt.xlabel('Energy [KeV]')
plt.ylabel('Efficiency')
plt.legend()
plt.savefig('Efficiency.pdf')
plt.show()

with open('EfficiencyParamaters.txt', 'wb') as fp:
    pickle.dump((u.correlated_values(popt, pcov), u.correlated_values(popt3, pcov3), u.correlated_values(popt4, pcov4), efficiency_coeff), fp)
