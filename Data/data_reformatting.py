# A script to convert the data from Maestro to a panda compatable HDF5 format
#Author: Thomas Bailey
#Date Created: 20181114

import numpy as np
import pandas as pd
import glob
import sys
sys.path += ["C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_Miniproject"]
from project_library import *

#iterator of files to read
files = glob.iglob('C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_New/*.Spe')
#files = ['C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_New\\2long_background.Spe']

#loop over all files
for file_location in files:
    #find the name of the file to be used for the reformated file
    file_name = file_location.replace('C:/Users/Thomas/Google Drive/Oxford/2018_19/NP03/NP03_New\\', '').replace('.Spe', '')

    #prepare the panda data frame that data is to be entered into
    data_frame = pd.DataFrame(columns=['BinNumber', 'Counts'])
    metadata = {}

    #open the file
    file_contents = open(file_location)
    #loop over the lines of the file to process
    line_number = 0 #counter to keep track of current line being read
    bin_number = 0 #counter to keep track of current bin being read
    calibration_flag = False #becomes true if the next line countains the embedded calibration
    for line in file_contents:
        #remove white space from the ends of the lines
        line = line.strip()

        #if statements to only read relevant lines
        if line_number == 7:
            #line with the time stamp
            metadata['TimeStamp'] = line
        elif line_number == 9:
            #line with run time, real and live
            run_time = line.split(' ')
            metadata['LiveTime'] = int(run_time[0])
            metadata['RealTime'] = int(run_time[1])
        elif (line_number >= 12) and (line_number <= 2059):
            #the lines with the data
            data_frame = data_frame.append({'Counts': int(line), 'BinNumber': int(bin_number)}, ignore_index=True)
            bin_number += 1
        elif line == '$ENER_FIT:':
            #check to see if next line contains the calibration
            calibration_flag = True
        elif calibration_flag:
            calibration_flag = False
            #save the embedded calibration
            calibration = line.split(' ')
            metadata['Calibration'] = [float(calibration[1]), float(calibration[0])]

        line_number += 1 #iterate the line number counter

    #save the data to a hfd5 file
    h5store(file_name+'.h5', data_frame, **metadata)
