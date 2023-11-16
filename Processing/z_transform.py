# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
z transformation of the data!

based on our studies, z-transforms on any other feature than charge is unhelpful
but we give you the option to try the other features if you so please...
'''
#######################################################################
#######################################################################

from MasterProcessor import *
import scipy.stats as stats

def ztransform(dataframe,feature):
    transformedData = []
    for chan in roughChannels:
        tempdf = dataframe[(dataframe.ChannelIDL == chan[0]) & (dataframe.ChannelIDR == chan[1])]
        transformedData.extend(stats.zscore(tempdf[feature]))
            
    return np.array(transformedData)

### read in datafiles ###
trainingData = pd.read_csv(energycut_trainingFile)
testingData = pd.read_csv(energycut_testingFile)

### TRANSFORM THE DATA ###
trainingData['ChargeL_zscore'] = ztransform(trainingData,"ChargeL")
trainingData['ChargeR_zscore'] = ztransform(trainingData,"ChargeR")

testingData['ChargeL_zscore'] = ztransform(testingData,"ChargeL")
testingData['ChargeR_zscore'] = ztransform(testingData,"ChargeR")

### Rewrite file with transformed data ###
print("Saving z-transformed data to files: " + energycut_trainingFile + " & " + energycut_testingFile)
trainingData.to_csv(energycut_trainingFile,index = False)
testingData.to_csv(energycut_testingFile,index = False)


