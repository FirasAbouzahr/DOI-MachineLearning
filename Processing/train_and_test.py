# Firas Abouzahr 09-27-23

########################################################################
########################################################################
'''

Here we read in our raw data and process it into training and testing data using functions
from our Header script. We create new features for our data from raw data calculations such as weighted energy.

'''
#######################################################################
#######################################################################

import sys
sys.path.append('/Users/feef/DOI-ML/Headers')


from DOI_header import *
from analysis_header import *


# get our raw training and testing datasets

dir = '/Users/feef/DOI_Data/' # set the directory where the data can be found
roughness = 28 # set what roughness data we want to look at
number_to_train_with = 10000 # set number of datapoints from EACH DOI to train the algorithim with, so 50000 yields a size of 7*50000 datapoints to train with
number_to_test_with = 2000 # set number of datapoints from EACH DOI to test the algorithim with


# lists to append each DOI dataframe to
trainingDataList = []
testingDataList = []


# Here we parse through the datasets for each of the 7 DOIs and sample the specifified number of datapoints for training & testing
for depth in DOIs:
    data = getDOIDataFrame(dir+'{}um_DOI_{}mm_coinc.txt'.format(roughness,depth),DOI=depth)
    trainingDataList.append(training)
    testingDataList.append(testing)
    

# concatenate our list of dataframes per DOI into one large dataframe & reset the index (for both testing and training data)
trainingData = pd.concat(trainingDataList,ignore_index=True)
testingData = pd.concat(testingDataList,ignore_index=True)


# calculating additional features that we calculate directly from the observables
# the power of NCD, normalized count differences, is discussed in the report and why we know to add this quantity is useful will be appear clear after training
trainingData["NCD"] = getNCD(trainingData.ChargeL,trainingData.ChargeR)
testingData["NCD"] = getNCD(testingData.ChargeL,testingData.ChargeR)
trainingData["delta_t"] = trainingData.TimeL - trainingData.TimeR
testingData["delta_t"] = testingData.TimeL - testingData.TimeR


# well need to read these files in again in other scripts so might as well make these into variables
trainingFile = 'trainingdata_{}um_channelpair{}-{}.csv'.format(roughness,channelpair[0],channelpair[1])
testingFile = 'testingdata_{}um_channelpair{}-{}.csv'.format(roughness,channelpair[0],channelpair[1])

trainingData.to_csv(trainingFile,index=False)
testingData.to_csv(testingFile,index=False)
