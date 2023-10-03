# Machine Learning to identify depth-of-interaction
Depth-of-interaction (DOI) is the depth at which a gamma-ray deposits its energy in a scintilator. DOI can significantly improve PET imaging by correcting parallax
error, the mispositioning of tomographic lines that connect two back-to-back gammas from an annihilation event. Without DOI information, we may assume that each gamma 
interacted at the center of each of the crystals in which they deposited their energy. Hence, when we draw the LOR between the two gammas, we connect the centers of the crystals.
However, oftentimes, the gammas will interact at different depths. For instance, one of the photons may deposit its energy at the front of a crystal while the other deposits 
its energy closer to the back of another crystal. In this case, naively drawing the tomographic line to connect the centers of the crystals would incorrectly localize where the annihilation event 
occurred. The importance of high-resolution image reconstruction calls for effective and efficient methods of determining DOI! Statistical analysis of experimental observables illustrate that
certain computed quantites change at different depths allowing us to sometimes distinguish between different DOIs and correct parallax error. Results from the analysis have already shown that DOI resolution, a measure of how well our detectors and corresponding analysis can identify DOI, is a function of crystal surface roguhness (more on this later)! In order to optimize the ability and rapidity of 
identifying DOI, we employ machine learning.

This readme document will:
1. Walk through the basic statistical analysis we can conduct on experimental observables to distinguish DOIs and the general results as well as how to reproduce these results using the python scripts in the [Analysis](/Analysis/) directory. 
2. Illustrate different pre-processing, data transformations, and more that we conduct on our datasets before training an algorithim to identify DOI with the [pre-processing](/pre-processing/) directory.
3. Show how we can begin training algorithims on processed datasets, assess the performance of an algorithim, the importance of various features, and more. Here we will use the [Machine-learning](/Machine-Learning/) and [Assessment](/Assessment/) directories. 

Throughout this document, we will also show how to cohesively use all of these scripts together and execute the shell scripts to automate the entire process. For more technical details on the physics, analysis, machine learning, and more, see my associated report on this research: [I will link it here when its done!]. Each directory will also have its own documentation with some additional discussion and walk through.

# [Analysis](/Analysis/)
