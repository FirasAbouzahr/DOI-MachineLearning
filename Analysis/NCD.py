# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
Here we visualize normalized count difference spectrum for a given roughness.
This is our main motivation for employing ML! We see a distinct difference bewteen
DOIs' NCD gaussians and which suggests ML may be able to use this information to identify
DOI on a signal by signal basis.
'''
#######################################################################
#######################################################################

from MasterAnalysis import *
normalize_to_15 = True # shift the NCD spectra such that 15 mm DOI is centered at 0
save_results = True
display_NCDs = True
pp_sigma = 1.5

framelist = []
for depth in DOIs:
    data = getDOIDataFrame(dir+'{}um_DOI_{}mm_coinc.txt'.format(roughness,depth),DOI=depth)
    framelist.append(data[(data.ChannelIDL == channelpair[0])])


if normalize_to_15 == True:
    left15mm_energy = getCharge(framelist[3],channelpair[0],"left",15)
    right15mm_energy = getCharge(framelist[3],channelpair[1],"right",15)
    
    yL,photopeakDataL = photopeakFit(left15mm_energy,energy_bins,photopeakcut = pp_sigma)
    yR,photopeakDataR = photopeakFit(right15mm_energy,energy_bins,photopeakcut = pp_sigma)
    
    NCD15 = getNCD(photopeakDataL,photopeakDataR)
    NCD15y,_ = photopeakFit(NCD15,NCD_bins,std_guess=None,photopeakcut=0)
    normFactor = NCD15y[1] # grab the mean of the gaussian fit, we then shift all data by this factor such that 15 mm is centered at 0

else:
    normFactor = 0

# now we repeat the procedure seen under --if normalize_to_15 == True-- for all data and

if display_NCDs == True:
    fig,ax = plt.subplots(figsize = (10,7))

    # if we do want to see the NCD plot, define some lists that will help us make pretty plots
    colorList = ['blue','red','green','purple','firebrick','navy','orange']
    DOIList = ['2 mm','5 mm','10 mm','15 mm','20 mm','25 mm','28 mm']

for df,index in zip(framelist,range(len(colorList))):
    left_energy = getCharge(df,channelpair[0],"left",DOIs[index])
    right_energy = getCharge(df,channelpair[1],"right",DOIs[index])

    yL,photopeakDataL = photopeakFit(left_energy,energy_bins,photopeakcut = pp_sigma)
    yR,photopeakDataR = photopeakFit(right_energy,energy_bins,photopeakcut = pp_sigma)

    NCD = getNCD(photopeakDataL,photopeakDataR)
    NCDy,NCDpeak = photopeakFit(NCD,NCD_bins,std_guess=None,photopeakcut=0)

    if display_NCDs == True:
        plotting_bins = np.linspace(NCD_bins[1],NCD_bins[2],NCD_bins[0])
        binHeights,binEdges,_ = ax.hist(NCD-normFactor,bins = plotting_bins,color = colorList[index],alpha = 0.6,label = DOIList[index])
        binCenters = (binEdges[:-1] + binEdges[1:]) / 2
        xspace = np.linspace(min(binCenters),max(binCenters),500)
        ax.plot(xspace,gaussian(xspace,NCDy[0],NCDy[1]-normFactor,NCDy[2]),color = colorList[index])


if display_NCDs == True:
    ax.set_ylabel('Counts',fontsize = 18)
    ax.set_xlabel('Normalized Count Differences',fontsize = 18)
    ax.set_title('NCD at varying DOI with {} µm roughness \n for Channel Pair {},{}'.format(roughness,channelpair[0],channelpair[1]),fontsize = 18)
    ax.tick_params("x",labelsize = 15)
    ax.tick_params("y",labelsize = 15)
    plt.legend(title = 'DOIs',fontsize = 14,title_fontsize=15)
#    lims = list(ax.get_xlim())
#    plt.xlim(lims[0],0.37)
    plt.show()
