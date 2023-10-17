# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
For ease-of-use and efficiency as we move datafiles down the processing pipeline, we define some constants that are used across all scirpts here.
Some very minuit variables are not defined here but in individual scripts
 
More importantly, this allows many files to be self-sufficient.
'''
#######################################################################
#######################################################################

import sys
sys.path.append('/Users/feef/DOI-ML/Headers')

from DOI_header import *
from analysis_header import *

roughness = 28 # set what roughness data we want to look at
sigma = 2 # number sigma we want to cut on our photpeaks for our datasets, imposed by calling photopeakcut.py

dir = '/Users/feef/DOI_Data/' # set the directory where the data can be found

trainingFile = 'trainingdata_{}um.csv'.format(roughness)
testingFile = 'testingdata_{}um.csv'.format(roughness)

energycut_trainingFile = 'trainingdata_{}um_{}σ_cut.csv'.format(roughness,sigma)
energycut_testingFile = 'testingdata_{}um_{}σ_cut.csv'.format(roughness,sigma)


photopeakLUTFile = "photopeakLUT_{}um_{}σ.csv".format(roughness,sigma)
