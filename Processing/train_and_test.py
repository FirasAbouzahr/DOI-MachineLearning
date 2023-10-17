# Firas Abouzahr 09-27-23

########################################################################
########################################################################
'''

Here we read in our raw data and process it into training and testing data using functions
from our Header script. We create new features for our data from raw data calculations such as weighted energy.

'''
#######################################################################
#######################################################################

from MasterProcessor import *

number_to_train_with = 10000 # set number of datapoints from EACH DOI to train the algorithim with, so 50000 yields a size of 7*50000 datapoints to train with
number_to_test_with = 2000 # set number of datapoints from EACH DOI to test the algorithim with
shuffle = True # choose whether we want to shuffle our data so that we don't train on DOI in order


# lists to append each DOI dataframe to
trainingDataList = []
testingDataList = []


# Here we parse through the datasets for each of the 7 DOIs and sample the specifified number of datapoints for training & testing per DOI
for depth in DOIs:
    data = getDOIDataFrame(dir+'{}um_DOI_{}mm_coinc.txt'.format(roughness,depth),DOI=depth)
    training,testing = train_and_test(data,number_to_train_with,number_to_test_with)
    trainingDataList.append(training)
    testingDataList.append(testing)
    
# concatenate our list of dataframes per DOI into one large dataframe & reset the index (for both testing and training data)
trainingData = pd.concat(trainingDataList,ignore_index=True)
testingData = pd.concat(testingDataList,ignore_index=True)

# we define here the sample sizes of our training and testing data which
# this is also useufl for keeping track of how statistics are being reduced as we perform data cuts
trainingSampleSize = np.shape(trainingData)[0]
testingSampleSize = np.shape(testingData)[0]


# calculating additional features that we calculate directly from the observables
# the power of NCD, normalized count differences, is discussed in the report and why we know to add this quantity is useful will be appear clear after training
trainingData["NCD"] = getNCD(trainingData.ChargeL,trainingData.ChargeR)
testingData["NCD"] = getNCD(testingData.ChargeL,testingData.ChargeR)
trainingData["delta_t"] = trainingData.TimeL - trainingData.TimeR
testingData["delta_t"] = testingData.TimeL - testingData.TimeR

# although train_and_test shuffles each DOI dataset in time, we may also want to shuffle the data again so DOI values are not in order
if shuffle == True:
    trainingData = trainingData.sample(n=trainingSampleSize)
    testingData = testingData.sample(n=testingSampleSize)

trainingData.to_csv(trainingFile,index=False)
testingData.to_csv(testingFile,index=False)

print("Original Sample Sizes:\n","Training Set:",trainingSampleSize,"\n","Testing Set:",testingSampleSize)
