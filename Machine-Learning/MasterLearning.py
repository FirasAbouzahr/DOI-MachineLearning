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
import tensorflow as tf
from tensorflow.python.ops import resources
import tensorflow_decision_forests as tfdf
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

sys.path.append('/Users/feef/Documents/GitHub/DOI-MachineLearning/Headers')
from DOI_header import *
from analysis_header import *

roughness = 28

# we get the data files ready for training from the processing directory
processed_dir = "/Users/feef/Documents/GitHub/DOI-MachineLearning/Processing/"

trainingFile = "trainingdata_28um_2σ_cut.csv"
testingFile = "testingdata_28um_2σ_cut.csv"

regressionFile = "regressionResults_{}um.csv".format(roughness)

# set custom features to train with
# MUST INCLUDE DOI
features = ["NCD","ChargeR","ChargeL","ChargeR_zscore","ChargeL_zscore","delta_t","ChannelIDL","ChannelIDR",'DOI']

