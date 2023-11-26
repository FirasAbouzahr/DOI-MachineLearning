# Firas Abouzahr 09-28-23

########################################################################
########################################################################
'''
Here we define some of our technical functions related to the specifics of our readout-electronics and software.
Much of these technical functions are legacy code - all credit to Kyle Klein, Will Matava, and Chris Layden
We also define our data read-in functions with pandas some constants that will be used throughout this repo
'''
#######################################################################
#######################################################################

import pandas as pd
import numpy as np
from tqdm import tqdm

#converts PETSys ID to geometric ID ; the d
def toGeo(x):
    y = 8*indices.get(x)[0] + indices.get(x)[1]
    return y

# this seems hard-coded but it corresponds to an interal system we have to go between the default channelIDs given in the raw data to our method of IDing pixels
indices = {
      0 : (4,7-7),
      1 : (4,7-6),
      2 : (7,7-5),
      3 : (5,7-7),
      4 : (5,7-4),
      5 : (5,7-5),
      6 : (4,7-4),
      7 : (7,7-7),
      8 : (6,7-6),
      9 : (7,7-4),
      10 : (5,7-6),
      11 : (6,7-4),
      12 : (4,7-5),
      13 : (6,7-5),
      14 : (6,7-7),
      15 : (7,7-6),
      16 : (3,7-7),
      17 : (3,7-6),
      18 : (2,7-7),
      19 : (2,7-6),
      20 : (0,7-7),
      21 : (1,7-7),
      22 : (0,7-6),
      23 : (1,7-6),
      24 : (3,7-5),
      25 : (1,7-5),
      26 : (2,7-5),
      27 : (4,7-3),
      28 : (0,7-5),
      29 : (3,7-4),
      30 : (0,7-4),
      31 : (1,7-4),
      32 : (2,7-4),
      33 : (3,7-3),
      34 : (2,7-3),
      35 : (0,7-3),
      36 : (1,7-3),
      37 : (0,7-2),
      38 : (5,7-3),
      39 : (1,7-2),
      40 : (2,7-2),
      41 : (3,7-2),
      42 : (1,7-1),
      43 : (0,7-1),
      44 : (0,7-0),
      45 : (3,7-1),
      46 : (1,7-0),
      47 : (2,7-1),
      48 : (3,7-0),
      49 : (2,7-0),
      50 : (6,7-2),
      51 : (6,7-1),
      52 : (7,7-1),
      53 : (4,7-1),
      54 : (5,7-1),
      55 : (6,7-0),
      56 : (7,7-0),
      57 : (7,7-2),
      58 : (7,7-3),
      59 : (4,7-2),
      60 : (5,7-0),
      61 : (5,7-2),
      62 : (6,7-3),
      63 : (4,7-0),
      64:(3+8,7),
      65:(3+8,6),
      66:(2+8,4),
      67:(2+8,6),
      68:(3+8,4),
      69:(1+8,7),
      70:(1+8,5),
      71:(0+8,7),
      72:(1+8,6),
      73:(3+8,3),
      74:(2+8,7),
      75:(2+8,3),
      76:(3+8,5),
      77:(0+8,5),
      78:(2+8,5),
      79:(0+8,6),
      80:(4+8,7),
      81:(6+8,7),
      82:(5+8,7),
      83:(7+8,7),
      84:(5+8,6),
      85:(4+8,6),
      86:(6+8,6),
      87:(7+8,6),
      88:(4+8,5),
      89:(6+8,5),
      90:(5+8,5),
      91:(1+8,4),
      92:(7+8,5),
      93:(7+8,4),
      94:(6+8,4),
      95:(4+8,4),
      96:(5+8,4),
      97:(5+8,3),
      98:(6+8,3),
      99:(4+8,3),
      100:(7+8,3),
      101:(7+8,2),
      102:(0+8,4),
      103:(6+8,2),
      104:(7+8,1),
      105:(5+8,2),
      106:(6+8,1),
      107:(4+8,2),
      108:(7+8,0),
      109:(5+8,1),
      110:(6+8,0),
      111:(4+8,1),
      112:(5+8,0),
      113:(4+8,0),
      114:(0+8,2),
      115:(2+8,1),
      116:(0+8,1),
      117:(3+8,1),
      118:(1+8,1),
      119:(1+8,0),
      120:(0+8,0),
      121:(1+8,2),
      122:(1+8,3),
      123:(3+8,2),
      124:(2+8,0),
      125:(2+8,2),
      126:(0+8,3),
      127:(3+8,0)}

geo_channels = []
for i in range(128):
    geo_channels.append([i,toGeo(i)])
geo_channels = np.asarray(geo_channels)
    
def toGeoChannelID(x):
    x = x % 128
    y = 8 * indices.get(x)[0] + indices.get(x)[1]
    return y
    
    
# the depths at which we ran our experiments at
DOIs = [2,5,10,15,20,25,28]

# channels that were coupled to rough crystals - for more context see Methods/Experimental Setup in the report
roughChannels = np.array([[ 78,  73],
                 [ 79,  72],
                 [ 86,  81],
                 [ 87,  80],
                 [ 94,  89],
                 [ 95,  88],
                 [102,  97],
                 [103,  96]])

# file = /path/to/datafile
# DOI = truth DOI from experiment
# toGeo = convert channelIDs into geometric IDs
# train_and_test = (T/F split into training & testing ; training size ; testing size)
def getDOIDataFrame(file,DOI):
    df = pd.read_csv(file, delimiter="\t", usecols=(2,3,4,12,13,14))
    df.columns = ["TimeL", "ChargeL", "ChannelIDL", "TimeR", "ChargeR", "ChannelIDR"]
    df["DOI"] = np.array([DOI]*np.shape(df)[0])
    
    df['ChannelIDL'] = df['ChannelIDL'].apply(toGeoChannelID)
    df['ChannelIDR'] = df['ChannelIDR'].apply(toGeoChannelID)
    
    df = df[(df.ChannelIDL.isin(roughChannels[:,0])) & (df.ChannelIDR.isin(roughChannels[:,1]))] # only keep channels whose data falls into rough crystal channels
    
    # reset index now that we've dropped the uninteresting rows of data
    df.index = np.arange(0,np.shape(df)[0],1)
    
    return df

# split data into training and testing and shuffle the data so we do not sample chronologically
def train_and_test(df,trainingsize,testingsize,shuffle = True):
    if shuffle == True:
        trainingData = df.sample(n=trainingsize)
        trainingIndices = np.where(df.index.isin(trainingData.index))[0]
        tempframe = df[~df.index.isin(trainingIndices)] # ensure we do not resample testing data from the same data used for training!
        testingData = tempframe.sample(n=testingsize)
    else:
        trainingData = df[df.index <= trainingsize]
        testingData = df[(df.index >= trainingsize) & (df.index <= testingsize)]
        
    return trainingData,testingData


