# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
Physics Motivation:
When a gamma enters into a scintillator it will either undergo photoelectric absorption, where the gamma fully deposits its 511 keV energy into the crystal,
or a compton scatter, where it scatters off an electron and only deposits some of its energy. In an energy spectrum, photoelectric absorption creates a photopeak,
a guassian peak center around 511 keV. The compton scatters manifest as a continuous shoulder to the left of the peak. Imposing a photopeak cut, only keeping datapoints
whose energies fall within some number of standard deviations within the photopeak. This ensures we do not take too much data from the compton shoulder, which blurs
the data and makes it difficult to reconstruct the line-of-repsonse.

DOI Motivation:
Moreover, for DOI, where location of the photopeak is one of the key factors to distinguish between different depths, isolating data in the photopeak is essential.
This is because while the photopeaks of each SiPM readout are typically separated, the compton shoulders will at least partially overlap. Hence, performing a
photopeak/energy cut on our data helps our algorithim better identify DOI.
'''
#######################################################################
#######################################################################

from MasterProcessor import *

use_saved_photopeakdata = True
overwrite_files = False

# no need to do anything but read these files since by this stage the data has already been formatted in train_and_test.py
trainingData = pd.read_csv(trainingFile)
testingData = pd.read_csv(testingFile)

# parse through all detector channels and fit to their photopeaks using getEnergySpectrum()
# then use the fit parameters and specified sigma to create a reference dataframe of photopeak cuts per channel per DOI
def getphotopeakcuts(df,sigma=2,energy_bins = (100,0,40),save_to_file = (False,"photopeaksheet_{}um.csv".format(roughness))): # roughness is defined in train_and_test.py
    photopeakDict = {'ChannelID':[],'energyCutLower':[],'energyCutUpper':[],'DOI':[]}
    
    for chanL in tqdm(np.unique(df.ChannelIDL)):
        for depth in DOIs:
            tempdf = df[df.DOI == depth]
            p,_ = getEnergySpectrum(tempdf,chanL,'left',bins=energy_bins)
            photopeakDict['ChannelID'] += [chanL]
            photopeakDict['energyCutLower'] += [p[1]-sigma*p[2]]
            photopeakDict['energyCutUpper'] += [p[1]+sigma*p[2]]
            photopeakDict['DOI'] += [depth]
            
    for chanR in tqdm(np.unique(df.ChannelIDR)):
        for depth in DOIs:
            tempdf = df[df.DOI == depth]
            p,_ = getEnergySpectrum(tempdf,chanR,'right',bins=energy_bins)
            photopeakDict['ChannelID'] += [chanR]
            photopeakDict['energyCutLower'] += [p[1]-sigma*p[2]]
            photopeakDict['energyCutUpper'] += [p[1]+sigma*p[2]]
            photopeakDict['DOI'] += [depth]

    photopeakDf = pd.DataFrame(photopeakDict)
    
    if save_to_file[0] == True:
        photopeakDf.to_csv(save_to_file[1],index = False)
    
    return photopeakDf

# using a photopeakcut look-up-table (LUT) that we generated using getphotopeakcuts(), we drop data that do not fall within some specified number of standard deviations aways from the photopeak center
def energycut(df,photopeakDf,filename):
    
    dfarray = df.to_numpy()
    energyCutDf = pd.DataFrame(columns = df.columns)
    index = 0

    for rownum in tqdm(range(len(dfarray))):
        row = dfarray[rownum]
        
        # get the energy cut for each channel at the given DOI
        ch1_lower_limit = photopeakDf[(photopeakDf.ChannelID == row[2]) & (photopeakDf.DOI == row[6])].energyCutLower.iloc[0]
        ch1_upper_limit = photopeakDf[(photopeakDf.ChannelID == row[2]) & (photopeakDf.DOI == row[6])].energyCutUpper.iloc[0]

        ch2_lower_limit = photopeakDf[(photopeakDf.ChannelID == row[5]) & (photopeakDf.DOI == row[6])].energyCutLower.iloc[0]
        ch2_upper_limit = photopeakDf[(photopeakDf.ChannelID == row[5]) & (photopeakDf.DOI == row[6])].energyCutUpper.iloc[0]
        energy1,energy2 = row[1],row[4]
        
        # only keep LORs whose energy values fall within the specified standard deviation of the photopeak mean
        if energy1>=ch1_lower_limit and energy1 <=ch1_upper_limit and energy2>=ch2_lower_limit and energy2 <=ch2_upper_limit:
                energyCutDf.loc[index] = row
                index += 1
    
    energyCutDf.to_csv(filename,index = False)
    
    return np.shape(energyCutDf)[0] # no need to return the frame, we just need it to save it to a file for later use

# generating or reading in an already saved photopeakcut LUT for the given roughness
try:
    print("Reading-in the {} µm,{}σ photopeak LUT".format(roughness,sigma))
    photopeakdata = pd.read_csv(photopeakLUTFile)
except:
    print("A photopeak LUT has yet to generated for {} µm data with a {}σ cut. \n Generating and saving the {} µm,{}σ LUT:".format(roughness,sigma,roughness,sigma))
    photopeakdata = getphotopeakcuts(trainingData,sigma,save_to_file = (True,photopeakLUTFile))

# here we call energycut() to impose energy cuts on both the training and testing data
if overwrite_files == True:
    newTrainingSize = energycut(trainingData,photopeakdata,trainingFile)
    newTestingSize = energycut(testingData,photopeakdata,testingFile)
else:
    newTrainingSize = energycut(trainingData,photopeakdata,energycut_trainingFile)
    newTestingSize = energycut(testingData,photopeakdata,energycut_testingFile)

print("Sample Sizes after the {}σ cut:\n".format(sigma),"Training Set:",newTrainingSize,"\n","Testing Set:",newTestingSize)
