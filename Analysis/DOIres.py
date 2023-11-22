# Firas Abouzahr 10-03-23

########################################################################
########################################################################
'''
Calculating resolutions from our NCD spectra plots!
'''
#######################################################################
#######################################################################

from MasterAnalysis import *
example_calculation = True


LUT = pd.read_csv("NCD_LUT_{}um.csv".format(roughness))
LUT = LUT[LUT.IDL == channelpair[0]]
print(LUT)
NCDinfo = LUT.to_numpy()[0][2:]


# some clever list comprehension (:
NCDpeaks = np.array([NCDinfo[evens] for evens in range(len(NCDinfo)) if evens%2 == 0])
NCDfwhms = np.array([NCDinfo[odds] for odds in range(len(NCDinfo)) if odds%2 != 0])

fig,ax = plt.subplots()
ax.scatter(NCDpeaks,DOIs)
ax.errorbar(NCDpeaks,DOIs,xerr = NCDfwhms/4,fmt = "none")
p,c = curve_fit(linear,NCDpeaks,DOIs)
x = np.linspace(min(NCDpeaks)-.01,max(NCDpeaks)+.01,100)
ax.plot(x,linear(x,*p),linestyle = "dashed",label = "Linear Fit")
lims = list(ax.get_xlim())


if example_calculation == True:
    exampleMean = LUT["{}mm".format(at_DOI)].iloc[0]
    exampleFWHM = LUT["{}mm_FWHM".format(at_DOI)].iloc[0]
    
    xmin = exampleMean - exampleFWHM/4
    xmax = exampleMean + exampleFWHM/4
    
    ymin = linear(xmin,p[0],p[1])
    ymax = linear(xmax,p[0],p[1])
    
    res = ymax - ymin
    
    ax.vlines(xmin,at_DOI,ymin,color = 'red')
    ax.vlines(xmax,at_DOI,ymax,color = 'red')
    ax.hlines(ymin,lims[0],xmin,color = 'red')
    ax.hlines(ymax,lims[0],xmax,color = 'red',label = 'Resolution Estimation')
    ax.text(-.25,at_DOI+.3,"Resolution at {} mm is {} mm".format(at_DOI,np.round(res,2)),color='red')
    
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.xlabel("Mean NCD",fontsize = 17)
    plt.ylabel("DOI [mm]",fontsize = 17)


ax.set_xlim(lims[0],lims[1])
plt.legend()
plt.show()




#def getResolution(fit_params,):
#/Users/feef/.Trash/Archive-Feb-11/DOI-paper-2023-shell-Feb-8-2023.pdf/Users/feef/.Trash/Archive-Feb-11/DOI-paper-2023-shell-Feb-8-2023.pdf
#    xmin = y - x/2
#    xmax = y + x/2
#
#    ymin = line(xmin,p[0],p[1])
#    ymax = line(xmax,p[0],p[1])
#
