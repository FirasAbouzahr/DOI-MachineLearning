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
pp_sigma = 1

NCD_frame = pd.DataFrame(columns = columns)
LUTindex = 0

for channelpair in tqdm(roughChannels):
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

    # now we repeat the procedure seen under --if normalize_to_15 == True-- for all data and grab means and FWHMs
    
    temporary_list = [channelpair[0],channelpair[1]]
    for df,index in zip(framelist,range(len(DOIs))):
        left_energy = getCharge(df,channelpair[0],"left",DOIs[index])
        right_energy = getCharge(df,channelpair[1],"right",DOIs[index])

        yL,photopeakDataL = photopeakFit(left_energy,energy_bins,photopeakcut = pp_sigma)
        yR,photopeakDataR = photopeakFit(right_energy,energy_bins,photopeakcut = pp_sigma)

        NCD = getNCD(photopeakDataL,photopeakDataR)
        NCDy,NCDpeak = photopeakFit(NCD,NCD_bins,std_guess=None,photopeakcut=0)
        
        temporary_list.append(NCDy[1] - normFactor)
        temporary_list.append(2.355*NCDy[2])
    
    NCD_frame.loc[LUTindex] = temporary_list
    LUTindex += 1
    
NCD_frame.to_csv(NCD_LUT,index = False)
