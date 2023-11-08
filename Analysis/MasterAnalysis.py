# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''

Here we define our reoccuring variables used throughout this analysis directory
and call to our header files.

'''
#######################################################################
#######################################################################

import sys
sys.path.append('/Users/feef/Documents/GitHub/DOI-MachineLearning/Headers')
from DOI_header import *
from analysis_header import *

dir = '/Users/feef/DOI_Data/'

# general variables used throughout analysis directory
roughness = 28
channelpair = roughChannels[1] # choose a channel to examine, the scripts will also allow for looks at all DOI channels at once.
energy_bins = (80,0,35) # (number of bins, lower limit, upper limit) this follows the syntax for np.histogram bins
NCD_bins = (100,-0.37,.41) 
at_DOI = 20 # this is specifically for energy-response plots, we look at each energy spectra on a DOI by DOI basis

# specific for getting NCD and calculating DOI resolutions
NCD_LUT = 'NCD_LUT_{}um.csv'.format(roughness) # filename for NCD.py to generate all datapoints and FWHM error bars
columns = ["IDL","IDR",
        "2mm","2mm_FWHM",
        "5mm","5mm_FWHM",
        "10mm","10mm_FWHM",
        "15mm","15mm_FWHM",
        "20mm","20mm_FWHM",
        "25mm","25mm_FWHM",
        "28mm","28mm_FWHM"]
