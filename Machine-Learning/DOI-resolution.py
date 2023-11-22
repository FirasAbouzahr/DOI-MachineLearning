# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
Obtaining DOI resolution and efficiency for every DOI at every channel
'''
#######################################################################
#######################################################################
import sys
from MasterLearning import *

omit_2_28 = True
print_averages = True
save_resolutions = True


if omit_2_28 == True: # necessary due to the asymmetric distributions
    useDOIs = DOIs[1:6]
    
else:
    useDOIs = DOIs
    
try:
    regressionResults = pd.read_csv("regressionResults_28um.csv")
    N = np.shape(regressionResults)[0]
except:
    print("Could not find the regression results. \nPlease run RFregression.py first to train and test the regression model. \nExiting...\n")
    sys.exit()

resolutionFrame = pd.DataFrame(columns = ["ChannelIDL","ChannelIDR","DOI","FWHM","error","efficiency"])
currentIndex = 0
for channelL,channelR in zip(roughChannels[:,0],roughChannels[:,1]):
    for depth in useDOIs:
        
        # get all instances at specific channel and DOI
        whereDOI = np.where((regressionResults.ChannelIDL == channelL) & (regressionResults.Truth == depth))[0]

        data = regressionResults.Truth.to_numpy()[whereDOI] - regressionResults.Predicted.to_numpy()[whereDOI]

        y,x = np.histogram(data,50)
        centers = (x[:-1] + x[1:]) / 2
        try:
            
            # construct a reasonable guess for our fit
            a = np.where(y == max(y))[0][0]
            mu = centers[a]
            A = y[a]
            std = 1
            guess = [A,mu,std]
            
            # fit to Gaussian to DOI
            p,c = curve_fit(gaussian,centers,y,p0 = guess)
            p = abs(p)
                       
            FWHM = p[2]*2.355
            ERR = np.sqrt(c[2,2])*2.355/np.sqrt(N) # standard error calculation
            
            # calculating efficiency
            inside = np.where((centers <= p[1] + 2.5*p[2]) & (centers >= p[1] - 2.5*p[2]))
            outside = np.where((centers >= p[1] + 2.5*p[2]) | (centers <= p[1] - 2.5*p[2]))
            yinside = y[inside]
            youtside = y[outside]
            
            efficiency = 1 - np.sum(youtside)/(np.sum(youtside) + np.sum(yinside))
            
            resolutionFrame.loc[currentIndex] = [channelL,channelR,depth,FWHM,ERR,efficiency]
            
            currentIndex += 1
        except:
            pass

if print_averages == True:
    print()
    if omit_2_28 == True:
        for depth in useDOIs:
            res = np.round(resolutionFrame[resolutionFrame.DOI == depth].FWHM.mean(),3)
            eff = np.round(resolutionFrame[resolutionFrame.DOI == depth].efficiency.mean(),3)
            print("Average Resolution at {} mm DOI: {} mm with {}% effciency".format(depth,res,eff*100))
