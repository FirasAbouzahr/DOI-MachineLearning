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

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from DOI_header import *


def gaussian(x,A,mu,sig):
    return A * np.exp(-((x-mu)/sig)**2)
    
def linear(x,m,b):
    return m * x + b
    
#def curvefit()

# guessing the standard deviation helps isolate the photopeak before we fit to it
def getEnergySpectrum(df,channelID,side,bins,std_guess = 2,photopeakcut = 1.5,display = False):
    fig,ax = plt.subplots()
       
    if side == 'left':
        df_by_chan = df[df.ChannelIDL == channelID]
        energy = df_by_chan.ChargeL
    
    else:
        df_by_chan = df[df.ChannelIDR == channelID]
        energy = df_by_chan.ChargeR
    
    y,x = np.histogram(energy,bins[0],(bins[1],bins[2]))
    centers = (x[:-1] + x[1:]) / 2
    
    # this helps isolate the photopeak making it easier for the fitter to find it
    fitcut = centers[np.where(y == max(y))[0][0]] - std_guess
    
    energy_temp = energy[energy >= fitcut]
    
    # redefine as we will fit to this cut data
    y,x = np.histogram(energy_temp,bins[0],(bins[1],bins[2]))
    centers = (x[:-1] + x[1:]) / 2
    

    guess = [max(y),centers[np.where(y == max(y))[0][0]],np.std(energy_temp)]
        
    try:
        p,c = curve_fit(gaussian,centers,y,p0=guess)
        xspace = np.linspace(p[1]-2.5*p[2],p[1]+2.5*p[2],500)
        ax.plot(xspace,gaussian(xspace,*p),color = 'red')
        ax.hist(energy,bins = np.linspace(bins[1],bins[2],bins[0]),color = 'C0')
        photopeak_counts = energy[(energy >= p[1] - photopeakcut*p[2]) & (energy <= p[1] + photopeakcut*p[2])]
    
    except:
        p = [-1,-1,-1]
        photopeak_counts = np.array([])
        print('Fit Failed')
        
    if display == False:
        plt.close()
        
    return p,photopeak_counts

# normalized count differences is a common metric used for DOI identification
def getNCD(left_Signal,right_signal):
    NCD = (left_Signal - right_signal)/((left_Signal + right_signal))
    return NCD



