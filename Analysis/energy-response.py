# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
Here we visualize a selected energy spectrum
'''
#######################################################################
#######################################################################

from MasterAnalysis import *

file = "/Users/feef/Documents/GitHub/DOI-MachineLearning/Processing/trainingdata_28um.csv"
df = pd.read_csv(file)

leftEnergy = getCharge(df,channelpair[0],'left',at_DOI)
rightEnergy = getCharge(df,channelpair[1],'right',at_DOI)

fig,ax = plt.subplots()
pL = plotEnergySpectrum(leftEnergy,energy_bins,fig,ax,label = 'Channel {}'.format(channelpair[0]),return_fit = True)
pR = plotEnergySpectrum(rightEnergy,energy_bins,fig,ax,label = 'Channel {}'.format(channelpair[1]),return_fit = True)
plt.legend()
plt.title("Coincidence Energy Spectra for Channel Pair {},{} \n at {} mm DOI".format(channelpair[0],channelpair[1],at_DOI))

leftCounts = len(leftEnergy)
leftPPmean = np.round(pL[1],2)
leftPPresolution = np.round(pL[2]*2.355/pL[1]*100,1)

rightCounts = len(rightEnergy)
rightPPmean = np.round(pR[1],2)
rightPPresolution = np.round(pR[2]*2.355/pR[1]*100,1)

plt.text(0.02,0.96,"Channel {}".format(channelpair[0]),transform = ax.transAxes)
plt.text(0.02,0.92,"Counts = {}".format(leftCounts),transform = ax.transAxes)
plt.text(0.02,0.88,"PP mean = {}".format(leftPPmean),transform = ax.transAxes)
plt.text(0.02,0.84,"Energy Res = {}%".format(leftPPresolution),transform = ax.transAxes)

plt.text(0.02,0.78,"Channel {}".format(channelpair[1]),transform = ax.transAxes)
plt.text(0.02,0.74,"Counts = {}".format(rightCounts),transform = ax.transAxes)
plt.text(0.02,0.7,"PP mean = {}".format(rightPPmean),transform = ax.transAxes)
plt.text(0.02,0.66,"Energy Res = {}%".format(rightPPresolution),transform = ax.transAxes)

plt.show()

