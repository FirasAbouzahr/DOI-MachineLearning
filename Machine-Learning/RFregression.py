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
from MasterLearning import *
from RFassessment import * 

prediction_spectra = True
save_results = True
feature_importance = True


trainingData = pd.read_csv(processed_dir + trainingFile)
testingData = pd.read_csv(processed_dir + testingFile)


try:
    kerasTrainingFrame_Regression = tfdf.keras.pd_dataframe_to_tf_dataset(trainingData[features], label="DOI",task=tfdf.keras.Task.REGRESSION)
    kerasTestingFrame_Regression = tfdf.keras.pd_dataframe_to_tf_dataset(testingData[features], label="DOI",task=tfdf.keras.Task.REGRESSION)
except:
    print("Please check the features selected in MasterLearning.py")
    print("Exiting...")
    sys.exit()

### TRAINING ###
print("Training the model:")
model_regression = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION,max_depth = 20)
model_regression.fit(kerasTrainingFrame_Regression)

### EVALUATING ###
print("\nEvaluating our model:")
model_regression.compile(metrics=["mse"])
evaluation = model_regression.evaluate(kerasTestingFrame_Regression, return_dict=True)

print(evaluation)
print()
MSE = evaluation['mse']
RMSE = np.sqrt(evaluation['mse'])
print("MSE: {}".format(MSE))
print("RMSE: {}".format(RMSE))

### TESTING ###
print("\nTesting our model:")
prediction = model_regression.predict(kerasTestingFrame_Regression)

# saving our results to a dataframe to optionally be readout
resultFrame = pd.DataFrame(columns = ["ChannelIDL","ChannelIDR","Truth","Predicted"])
resultFrame["ChannelIDL"] = testingData.ChannelIDL
resultFrame["ChannelIDR"] = testingData.ChannelIDR
resultFrame["Truth"] = testingData.DOI
resultFrame["Predicted"] = prediction

### ASSESSING ###

# if we don't want to visualize this right away, we can plot this later using saved regression results
if prediction_spectra == True:
    predictionSpectra(resultFrame["Predicted"],resultFrame["Truth"],DOIs = DOIs)
    
if feature_importance == True:
    feat_importance = getFeatureImportance(model_regression,metric = 'INV_MEAN_MIN_DEPTH',display = True)
    
if save_results == True:
    print("\nSaving regression results to file: " + regressionFile)
    resultFrame.to_csv(regressionFile,index = False)
