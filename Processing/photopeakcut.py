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

import sys
sys.path.append('/Users/feef/DOI-ML/Headers')


from DOI_header import *
from analysis_header import *
from train_and_test import *

# no need to do anything but read these files since by this stage the data has already been formatted in train_and_test.py
trainingData = pd.read_csv(trainingFile)
testingData = pd.read_csv(testingFile)

# parse through all detector channels and fit to their photopeaks using getEnergySpectrum()
# then use the fit parameters and specified sigma to create a reference dataframe of photopeak cuts per channel per DOI
def getphotopeakcuts(df,sigma = 2,energy_bins = (100,0,40),save_to_file = (False,"photopeaksheet_{}um.csv".format(roughness))):
    photopeakDict = {'ChannelID':[],'energyCut':[],'DOI':[]}
    
    for chanL in tqdm(np.unique(df.ChannelIDL)):
        for depth in DOIs:
            tempdf = df[df.DOI == depth]
            p,_ = getEnergySpectrum(tempdf,chanL,'left',bins=energy_bins)
            photopeakDict['ChannelID'] += [chanL]
            photopeakDict['energyCut'] += [[p[1]-sigma*p[2],p[1]+sigma*p[2]]]
            photopeakDict['DOI'] += [depth]
            
    for chanR in tqdm(np.unique(df.ChannelIDR)):
        for depth in DOIs:
            tempdf = df[df.DOI == depth]
            p,_ = getEnergySpectrum(tempdf,chanR,'right',bins=energy_bins)
            photopeakDict['ChannelID'] += [chanR]
            photopeakDict['energyCut'] += [[p[1]-sigma*p[2],p[1]+sigma*p[2]]]
            photopeakDict['DOI'] += [depth]

    photopeakDf = pd.DataFrame(photopeakDict)
    
    if save_to_file[0] == True:
        photopeakDf.to_csv(save_to_file[1],index = False)
    
    return photopeakDf

# using a photopeakcut datasheet, we drop data do not fall within some specified number of standard deviations aways from the photopeak center
def energycut(df,photopeakDf,filename):
    
    dfarray = df.to_numpy()
    energyCutDf = pd.DataFrame(columns = df.columns)
    index = 0

    for rownum in tqdm(range(len(dfarray))):
        row = dfarray[rownum]
        
        # get the energy cut for each channel at the given DOI
        ch1_cut = photopeakDf[(photopeakDf.ChannelID == row[2]) & (photopeakDf.DOI == row[6])].energyCut.iloc[0]
        ch1_lower_limit,ch1_upper_limit = ch1_cut[0],ch1_cut[1]
        ch2_cut = photopeakDf[(photopeakDf.ChannelID == row[5]) & (photopeakDf.DOI == row[6])].energyCut.iloc[0]
        ch2_lower_limit,ch2_upper_limit = ch2_cut[0],ch2_cut[1]
        
        energy1,energy2 = row[1],row[4]
        
        # only keep LORs whose energy values fall within the specified standard deviation of the photopeak mean
        if energy1>=ch1_lower_limit and energy1 <=ch1_upper_limit and energy2>=ch2_lower_limit and energy2 <=ch2_upper_limit:
                energyCutDf.loc[index] = row
                index += 1
        
    energyCutDf.to_csv(filename,index = False)
    
    return 0


# lets generate our photopeak dataframe with 2 sigma photopeak cuts
# we only need to do this for the trainingData since the testingData should use the same channel pair(s)
photopeakdata = getphotopeakcuts(trainingData)
    
    
# here we call energycut() to impose energy cuts on both the training and testing data

trainingData_with_energycut = energycut(trainingData,photopeakdata,'trainingdata_{}um_channelpair{}-{}_energycut.csv'.format(roughness,channelpair[0],channelpair[1]))
testingData_with_energycut = energycut(testingData,photopeakdata,'testingdata_{}um_channelpair{}-{}_energycut.csv'.format(roughness,channelpair[0],channelpair[1]))
