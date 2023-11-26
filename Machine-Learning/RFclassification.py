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

feature_importance = True
confusion_matrix = True


trainingData = pd.read_csv(processed_dir + trainingFile)
testingData = pd.read_csv(processed_dir + testingFile)


try:
    kerasTrainingFrame = tfdf.keras.pd_dataframe_to_tf_dataset(trainingData[features], label="DOI")
    kerasPredictionFrame = tfdf.keras.pd_dataframe_to_tf_dataset(testingData[features], label="DOI")
except:
    print("Please check the features selected in MasterLearning.py")
    print("Exiting...")
    sys.exit()

### TRAINING ###
print("Training the model:")
model = tfdf.keras.RandomForestModel(max_depth = 20,verbose=2) # set parameters here
model.fit(kerasTrainingFrame,label='DOI')

### EVALUATING ###
print("\nEvaluating our model:")
compilation = model.compile(metrics=["Accuracy"])
evaluation = model.evaluate(kerasPredictionFrame,return_dict=True)

### TESTING ###
predict = model.predict(kerasPredictionFrame)
predict = np.argmax(predict, axis=1)
predict = predict.reshape(predict.shape[0], 1)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

### ASSESSING ###
if feature_importance == True:
    feat_importance = getFeatureImportance(model,metric = 'INV_MEAN_MIN_DEPTH',display = True)

if confusion_matrix:
    matrix = ConfusionMatrix(predict,testingData)
