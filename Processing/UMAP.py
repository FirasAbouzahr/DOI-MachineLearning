# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
UMAP PROJECTIONS!
'''
#######################################################################
#######################################################################

from MasterProcessor import *
import umap.umap_ as umap

show_projection = True
save_as_features = True

# dimensionality reductions should really be done with energy cut data!
# if desired to visualize clustering without cut data change: energycut_trainingFile --> trainingFile, and similar for testing
trainingData = pd.read_csv(energycut_trainingFile)
testingData = pd.read_csv(energycut_testingFile)

features = ["NCD","ChargeR","ChargeL","ChannelIDL","ChannelIDR"] # choose which features we want to UMAP project
output = ["DOI"]

xTrain = trainingData[features]
yTrain = trainingData[output]
xTest = testingData[features]
yTest = testingData[output]

# here we actually setup our UMAP model
print("\nFitting our data to ")
staticReduction = umap.UMAP(output_metric = "chebyshev",n_components=3,random_state=42,verbose=True).fit(xTrain)

### CHANGE MANUALLY ACCORDING TO THE DIMENSION OF PHASE SPACE WE WANT ###
n1_training = staticReduction.embedding_[:,0]
n2_training = staticReduction.embedding_[:,1]
n3_training = staticReduction.embedding_[:,2]

# enter variables as a list like variables = [n1,n2,n3]
def showClustering(variables,truthValues,dimensions = 2):
    
    DOIdensity = truthValues
    bounds = [2,5,10,15,20,25,28,30] #DOIs
    fakebounds = [2-1,5-2.5,10-2,15-2,20-2,25-2,28-1,30-1] # these hat maps are a bit annoying sometimes, heres a cute trick to create a discrete legend
    
    if dimensions == 3:
        fig = plt.figure(figsize = (9,9))
        ax = fig.add_subplot(projection='3d')
        scatter = plt.scatter(variables[0],variables[1],variables[2],c=DOIdensity,cmap='Spectral')
        ax.tick_params("z",labelsize = 13)
        
    elif dimensions == 2:
        fig,ax = plt.subplots(figsize = (9,9))
        scatter = plt.scatter(variables[0],variables[1],c=DOIdensity,cmap='Spectral')
        
    else:
        raise Exception("Humans can't visualize more than 3 dimensions at a time!")
        
    cbar = plt.colorbar(scatter,spacing='proportional', ticks=bounds, boundaries=np.array(fakebounds), format='%1i',label = 'DOI',fraction=0.03)
    ax.tick_params("y",labelsize = 13)
    ax.tick_params("x",labelsize = 13)
    plt.show()
        

# plot the projection to see what its doing
if show_projection == True:
    variables = [n1_training,n2_training,n3_training]
    truthValues = trainingData.DOI.to_numpy()
    showClustering(variables,truthValues,dimensions = 3)

print("\nCarrying out the same projection on the training data:")
testEmbeddings = staticReduction.transform(xTest)

n1_testing = testEmbeddings[:,0]
n2_testing = testEmbeddings[:,1]
n3_testing = testEmbeddings[:,2]

if save_as_features == True:
    try:
        trainingData["n1"] = n1_training
        trainingData["n2"] = n2_training
        trainingData["n3"] = n3_training
        
        testingData["n1"] = n1_testing
        testingData["n2"] = n2_testing
        testingData["n3"] = n3_testing
        
        trainingData.to_csv(energycut_trainingFile,index = False)
        testingData.to_csv(energycut_testingFile,index = False)
        
    except:
        print("\nCannot add features to datasets, please check number of dimensions and define or delete variables accordingly.")
