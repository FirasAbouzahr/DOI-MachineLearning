# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''

'''
#######################################################################
#######################################################################

from MasterLearning import *


def getFeatureImportance(model,metric = 'INV_MEAN_MIN_DEPTH',display = True):
    importances = (model.make_inspector().variable_importances()['INV_MEAN_MIN_DEPTH'])
    used_features = []
    feat_importance = []
    for feats in importances:
        used_features.append(feats[0][0])
        feat_importance.append(feats[1])

    if display == True:
        fig,ax = plt.subplots(figsize = (16,7))
        plt.bar(used_features,feat_importance)
        plt.title('Inverse Mean Minumum Depth per Feature',fontsize = 18)
        plt.ylabel('Inverse Mean Minimum Depth',fontsize = 18)
        plt.xlabel('Feature',fontsize = 18)
        plt.xticks(fontsize = 15,rotation = 45)
        plt.yticks(fontsize = 15)
        plt.show()
        
    return feat_importance


def ConfusionMatrix(predict,testingData,normalize_per_category = True,title = ""):

    # define our confusion matrix
    matrix = np.transpose(confusion_matrix(predict,testingData.DOI))

    if normalize_per_category == True:
        matrix = np.round(normalize(matrix,axis=1,norm='l1'),2) # so we deal with percentages rather than number correct/incorrect
        lowcolor = 0.7
    else:
        lowcolor = 4000

    DOI_index = 0
    correct = 0
    notCorrect = 0
    for row in matrix:
        accuracy = row[DOI_index]/np.sum(row)
        correct += row[DOI_index]
        notCorrect += np.sum(row) - row[DOI_index]
        print('Accuracy for ' + str(DOIs[DOI_index]) + ' mm DOI:',np.round(accuracy,3))
        DOI_index += 1
    print('Total Accuracy: ',np.round(correct/(correct + notCorrect),3))

    # imshow will plot this by index in the list
    doiIndices = np.arange(0,7,1)

    fig,ax = plt.subplots(figsize = (10,7))
    plt.imshow(matrix,cmap='Blues')
    plt.colorbar().set_label(label='Accuracy per Category',size=15)
    plt.xticks(doiIndices,DOIs,fontsize = 15)
    plt.yticks(doiIndices,DOIs,fontsize = 15)

    for i in doiIndices:
        for j in doiIndices:
            if matrix[j,i] < lowcolor:
                plt.text(i,j,matrix[j,i],ha="center", va="center",c='black',fontsize = 13)
            else:
                plt.text(i,j,matrix[j,i],ha="center", va="center",c='w',fontsize = 13)

    plt.ylabel('Truth DOI',fontsize = 18)
    plt.xlabel('Predicted DOI',fontsize = 18)
    plt.title(title,fontsize = 18)

    return matrix


def predictionSpectra(prediction,truth,DOIs): # enter DOIs as list even if a single DOI ~ like [2] for to look at the 2 mm spectrum
    fig,ax = plt.subplots(figsize = (10,7))
    newDOIs = DOIs[-1:]
    
    # loop through all DOIs and plot what was predicted for them by our regression model
    for depth in DOIs:
        whereDOI = np.where(truth == depth)[0]
        bins = np.linspace(depth-10,depth+10,200)
        y,x,_ = plt.hist(prediction[whereDOI],bins = bins,label = '{} mm'.format(depth))
    
    plt.ylabel('Counts',fontsize = 18)
    plt.xlabel('Predicted DOI [mm]',fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.title('RF Regression DOI Prediction Distributions',fontsize = 18)
    leg = plt.legend(title = "DOI",fontsize = 13)
    leg.set_title("DOI", prop = {'size':'x-large'})
    plt.show()
