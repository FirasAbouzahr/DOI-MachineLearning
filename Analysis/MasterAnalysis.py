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

roughness = 28
channelpair = roughChannels[3] # choose a channel to examine, the scripts will also allow for looks at all DOI channels at once.
energy_bins = (80,0,35) # (number of bins, lower limit, upper limit) this follows the syntax for np.histogram bins
NCD_bins = (100,-0.37,.41) 
at_DOI = 5 # this is specifically for energy-response plots, we look at each energy spectra on a DOI by DOI basis


