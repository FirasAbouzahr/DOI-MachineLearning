# Machine Learning to identify depth-of-interaction
Depth-of-interaction (DOI) is the depth at which a gamma-ray deposits its energy in a scintilator. DOI can significantly improve PET imaging by correcting parallax
error, the mispositioning of tomographic lines that connect two back-to-back gammas from an annihilation event. Without DOI information, we may assume that each gamma 
interacted at the center of each of the crystals in which they deposited their energy. Hence, when we draw the LOR between the two gammas, we connect the centers of the crystals.
However, oftentimes, the gammas will interact at different depths. For instance, one of the photons may deposit its energy at the front of a crystal while the other deposits 
its energy closer to the back of another crystal. In this case, naively drawing the tomographic line to connect the centers of the crystals would incorrectly localize where the annihilation event 
occurred. The importance of high-resolution image reconstruction calls for effective and efficient methods of determining DOI! Specialized PET scanners that utilize dual-ended readout, a PET scanner with SiPMs coupled on both sides of its scintillator arrays, can yield experimental observables that can allow us to identify DOI. Analysis of experimental data from dual-ended PET illustrates that
certain experimental quantites change as a function of depth allowing us to sometimes distinguish between different DOIs and correct parallax error. Results from the analysis have already shown that DOI resolution, a measure of how well our detectors and corresponding analysis can identify DOI, is a function of crystal surface roguhness (more on this later)! In order to optimize the ability and rapidity of 
identifying DOI, we employ machine learning.

# Data Processing
The data processing pipeline includes five python scripts. MasterProcessor.py, is imported into the four other scripts, and is simply used as a place to define certain reoccurring variables names like file names for which the processed data is saved under. This file also imports the Header files, DOI\_header.py and analysis\_header.py, which define various functions, such as the photopeak fitting function, used throughout the processing codes. For each DOI measured and for every crystal roughness type, there is a separate data file. The first step in the pipeline is to call train\_and\_test.py, which reads-in every DOI file and concatenates them into two datasets, a training and a testing set. At the beginning of train\_and\_test.py, we define three variables: number\_to\_train\_with, number\_to\_test\_with, and shuffle. The first two variables are integers and define how many data points we want to sample per DOI for the training and testing data. Shuffle is a boolean and when set to True, scrambles the created training and testing data. By reading in each DOI data file through a loop, we sample the specified number of data points for each DOI for training and testing and concatenate them into their own pandas DataFrames and then finally save them into their own files. \\ 

Before we call train\_and\_test.py, there are no data files in the Processing directory:

```
Processing Firas$ ls
MasterProcessor.py        clean.sh            readme.md
UMAP.py                 train_and_test.py      __init__.py            
photopeakcut.py            z_transform.py
```

Now, setting the three variables at the beginning of the script to number\_to\_train\_with = 50000, number\_to\_test\_with = 20000, and shuffle = True and calling train\_and\_test.py:

```
Processing Firas$ python3 train_and_test.py
Original Sample Sizes:
 Training Set: 350000 
 Testing Set: 140000
```

The script prints out the sample sizes of the training and testing datasets produced for reference. Now, in the directory we have two files, one for training (trainingdata\_28um.csv) and one for testing (testingdata\_28um.csv) our random forest: 

```
Processing Firas$ ls
MasterProcessor.py    train_and_test.py    UMAP.py                
photopeakcut.py        trainingdata_28um.csv
__init__.py            readme.md            z_transform.py
clean.sh            testingdata_28um.csv
Firass-MacBook-Pro:Processing Firas$ 
```

 

train\_and\_test.py also added two new columns to these data files not present in the original data sets, normalized count differences (NCD) and detection time difference (delta\_t). Other than these two newly defined variables, the files generated by this script have no filtering done to them. The next step in the pipeline is to conduct an energy cut on the datasets. To do this, we use photopeakcut.py, which has two main functions. The first function getphotopeakcuts() is used to generate a look-up-table (LUT) of photopeak cuts. getphotopeakcuts() loops through all of the SiPM channels present in our data files, finds and fits to their respective energy spectrum photopeaks at each DOI, and then saves the lower- and upper-limit photopeak cut for each photopeak. The number of standard deviations from the mean that the two limits correspond to is given by the input parameter sigma. Once a photopeak cut LUT has been generated, we can use the next function, energycut(), to actually impose photopeak cuts on our training and testing datasets. The script by default imposes the energy cut on trainingdata\_28um.csv and testingdata\_28um.csv generated by train\_and\_test.py. 

 

```
Processing Firas$ ls
MasterProcessor.py    train_and_test.py    UMAP.py                
photopeakcut.py        trainingdata_28um.csv
__init__.py            readme.md            z_transform.py
clean.sh            testingdata_28um.csv
```

 

I've included a couple nice features to make this as streamline as possible. Notice that the script tells you that no LUT was found and thus it must be generated and saved. Secondly, the use of the wonderful python package tqdm allows us to view a loading bar as we impose our energy cuts. After imposing the cut, two new files are generated trainingdata\_28um\_?$\sigma$\_cut.csv and testingdata\_28um\_?$\sigma$\_cut.csv, where the ? indicates the value of sigma. Below is an example of running photopeakcut.py with sigma=2.

 

```
Processing Firas$ python3 photopeakcut.py

Trying to read-in the 28 um,2sig photopeak LUT...

A photopeak LUT has yet to generated for 28 um data with a 2sig cut. 
Generating and saving the 28 um,2sig LUT:
100%|============================================| 8/8 [00:00<00:00, 92.49it/s]
100%|============================================| 8/8 [00:00<00:00, 95.25it/s]

Imposing a 2sig photopeak cut on training and testing datasets... this may take a few minutes!
100%|==================================================================| 350000/350000 [05:04<00:00, 1149.37it/s]
100%|==================================================================| 140000/140000 [01:46<00:00, 1310.79it/s]
Sample Sizes after the 2sig cut:
 Training Set: 93171 
 Testing Set: 37319
```

 

The tqdm loading bars couldn't be rendered in \LaTeX \: so I've replaced them with equal signs. It also did not like the $\sigma$ or $\mu$ being printed in the code box so I've replaced them with ``sig`` and ``u`` for the purpose of the report but normally, shell would print the appropriate symbols. Notice the script prints out the training and testing sample sizes after the cut. Now with our most significant data processing done, we can move onto ``optional`` data transformation scripts, namely using UMAP.py or z\_transform.py, both of which by default read-in the energy cut data (so trainingdata\_28um\_?$\sigma$\_cut.csv and testingdata\_28um\_?$\sigma$\_cut.csv) and then re-save the files with new columns of the transformed data. Of course, a simple change of variable name would allow us to also carry out the same transformations on unfiltered data. Before running any of our optional transformation scripts, our two energy cut files have the following columns: 

 

```
Processing Firas$ python3
Python 3.9.6 (default, May  7 2023, 23:32:44) 
[Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> df = pd.read_csv("trainingdata_28um_2σ_cut.csv")
>>> df.columns
Index(['TimeL', 'ChargeL', 'ChannelIDL', 'TimeR', 'ChargeR', 'ChannelIDR',
       'DOI', 'NCD', 'delta_t'],
      dtype='object')
```

 

Now, let's run UMAP.py. UMAP.py starts by allowing us to choose which features we want to cluster with UMAP, which by default is set to what was used in our report (and found to yield the best result): charge, NCD, and channelID. Then it actually calls upon umap.UMAP() given by the UMAP package with a few preset parameters: metric=``chebyshev`` and n\_components = 3, again the same parameters our studies illustrated to yield the best results (see Section UMAP in my thesis above). UMAP.py also gives one the option to plot the clustering with the boolean variable, show\_projection.
 

```
Processing Firas$ python3 UMAP.py
Fitting our data to
UMAP(n_components=3, output_metric='chebyshev', random_state=42, verbose=True)
Sun Nov 26 10:03:04 2023 Construct fuzzy simplicial set
Sun Nov 26 10:03:04 2023 Finding Nearest Neighbors
Sun Nov 26 10:03:04 2023 Building RP forest with 20 trees
Sun Nov 26 10:03:06 2023 NN descent for 17 iterations
     1  /  17
     2  /  17
    Stopping threshold met -- exiting after 2 iterations
Sun Nov 26 10:03:18 2023 Finished Nearest Neighbor Search
Sun Nov 26 10:03:20 2023 Construct embedding
Epochs completed: 100%|====================================================|200/200 [00:45]
Sun Nov 26 10:04:18 2023 Finished embedding

Carrying out the same projection on the training data:
Sun Nov 26 10:04:55 2023 Worst tree score: 0.99050134
Sun Nov 26 10:04:55 2023 Mean tree score: 0.99159985
Sun Nov 26 10:04:55 2023 Best tree score: 0.99242253
Sun Nov 26 10:04:56 2023 Forward diversification reduced edges from 1397565 to 379863
Sun Nov 26 10:04:57 2023 Reverse diversification reduced edges from 379863 to 379863
Sun Nov 26 10:04:58 2023 Degree pruning reduced edges from 327222 to 327222
Sun Nov 26 10:04:58 2023 Resorting data and graph based on tree order
Sun Nov 26 10:04:58 2023 Building and compiling search function
Epochs completed: 100%|====================================================|30/30 [00:02]
```

 

The text shown above are all a result of having verbose=True in umap.UMAP(). With show\_projection = True, we also see a resultant clustering plot similar to the one shown below.

![png](Figures/output_17_4.png)

Now after runnning UMAP.py, we have new transformed columns in our datasets: 

```
Processing Firas$ python3
Python 3.9.6 (default, May  7 2023, 23:32:44) 
[Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> df = pd.read_csv("trainingdata_28um_2σ_cut.csv")
>>> df.columns
Index(['TimeL', 'ChargeL', 'ChannelIDL', 'TimeR', 'ChargeR', 'ChannelIDR',
       'DOI', 'NCD', 'delta_t', 'n1', 'n2', 'n3'],
      dtype='object')
```

 

The n1, n2, n2 represent our new phase space variables from the projection. We can also use z\_transform.py to carry out a z-score transformation of our data. Using the function ztransform(), which itself calls upon scipy.stats.zscore, we can z-score transform any column of data we want from a pandas DataFrame \cite{pandas,SCIPY}. The script by default carries out this transform only on charge as this was proven to yield promising results. 

```
Processing Firas$ python3 z_transform.py
Saving z-transformed data to files: trainingdata_28um_2σ_cut.csv & testingdata_28um_2σ_cut.csv
```

 

And again, these transformed features are added to the energy-cut data files: 

 

```
Processing Firas$ python3
Python 3.9.6 (default, May  7 2023, 23:32:44) 
[Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> df = pd.read_csv("trainingdata_28um_2σ_cut.csv")
>>> df.columns
Index(['TimeL', 'ChargeL', 'ChannelIDL', 'TimeR', 'ChargeR', 'ChannelIDR',
       'DOI', 'NCD', 'delta_t', 'n1', 'n2', 'n3', 'ChargeL_zscore',
       'ChargeR_zscore'],
      dtype='object')
```

 

# Machine Learning Pipeline
Once, we have ran the required scripts in the Processing directory and, if desired, the two optional scripts, we can move onto machine learning! Since our machine learning is still at the research stage, the learning pipeline is set to train and test in the same scripts. Work is currently being done to streamline this directory such that saved random forest models can be applied to new datasets. For now the repository contains five main scripts. As with the Processing directory, this file has a ``Master`` file that only contains variables that will be reused throughout the ML pipeline such as the features we wish to implement into our models. Next, we have RFassessment.py, which contains functions to assess how our classification and regression random forest models are performing. This includes functions to plot the confusion matrix, prediction distributions, and feature importance calculations. RFassessment.py is not called on its own but rather is imported into our actual machine learning scripts: RFclassification and RFregression.py which both give options to show assessment plots after testing. RFclassification and RFregression.py actually train and test random forest models using the data files generated in the Processing directory with the features defined in MasterLearning.py. The last script, DOI-resolution.py, uses the results saved from RFregression.py to fit Gaussians to the truth minus predicted distributions and then extract DOI resolutions. Let's start with RFclassification. In MasterLearning.py we have set:

 

```python
features = ["NCD","ChargeR","ChargeL","ChargeR_zscore","ChargeL_zscore","delta_t","ChannelIDL","ChannelIDR",'DOI']
```

 

RFclassification.py first calls tfdf.keras.pd\_dataframe\_to\_tf\_dataset() to convert our training and testing data files from pandas DataFrames to keras frames, which is more optimized for use with TensorFlow. After these are defined, the script creates a random forest model using TensorFlow: tfdf.keras.RandomForestModel() with certain parameters set to default according to the results of this report. With the parameter verbose=2, while the model is being trained, many statements are printed out such as the accuracy per tree. After the model has been trained, the script calls the TensorFlow functions model.compile(), model.evaluate(), and model.predict() to evaluate our model and then use it to predict the DOIs from our testing data. Ignoring the verbosity output by TensorFlow, when we call RFclassification.py we see:


```
Machine-Learning Firas$ python3 RFclassification.py
Training the model:

.
. this is where TensorFlow would print out information about the RF being built
.

Evaluating our model:
36/36 [==============================] - 1s 27ms/step - loss: 0.0000e+00 - Accuracy: 0.8209
36/36 [==============================] - 1s 27ms/step
loss: 0.0000
Accuracy: 0.8209

Accuracy for 2 mm DOI: 0.707
Accuracy for 5 mm DOI: 0.717
Accuracy for 10 mm DOI: 0.92
Accuracy for 15 mm DOI: 0.74
Accuracy for 20 mm DOI: 0.85
Accuracy for 25 mm DOI: 0.87
Accuracy for 28 mm DOI: 0.96
Total Accuracy:  0.824
```

 

RFclassification has two boolean variables defined: feature\_importance and confusion\_matrix. When feature\_importance is set to True, the script calls the function getFeatureImportance() from RFassessment.py, which by default computes the mean inverse minimum depth, and will then show a plot like so (depending on which features we actually chose to train our RF model on):

![png](Figures/output_25_1.png)

With confusion\_matrix set to True, the script calls ConfusionMatrix() from RFassessment.py to plot a confusion matrix just like in Figures like below:

![png](Figures/output_28_2.png)

RFregression.py follows a very similar setup to RFclassification with task=tfdf.keras.Task.REGRESSION in the tfdf.keras.RandomForestModel() function to grow a regression random forest. An additional feature of this script is that it saves the prediction results to a file named regressionResults\_28um.csv to estimate DOI resolution later on. 

```
Machine-Learning Firas$ python3 RFregression.py
Training the model:

.
. this is where TensorFlow would print out information about the RF being built
.

Evaluating our model:
36/36 [==============================] - 1s 30ms/step - loss: 0.0000e+00 - mse: 1.9671
{'loss': 0.0, 'mse': 1.9671297073364258}

MSE: 1.9671297073364258
RMSE: 1.4025440126200768

Testing our model:
38/38 [==============================] - 2s 47ms/step

Saving regression results to file: regressionResults_28um.csv
```


As with RFclassification, RFregression also has two boolean variables we can set equal to True to help visualize the performance of our model. The first is feature\_importance, which serves the same purpose as in classification. The second is prediction\_spectra, which when set to True, will display a plot as below:

![png](Figures/output_35_1.png)

Finally, after running RFregression.py, we will have a file named regressionResults\_28um.csv. With this file generated, we can run DOI-resolution.py to estimate the average DOI resolutions per DOI. DOI-resolution.py has the boolean omit\_2\_28 = True because as noted in the report, for now, we do not cite the 2 and 28 mm DOI resolutions due to the edge effects.

 
```
Machine-Learning Firas$ python3 DOI-resolution.py

Average Resolution at 5 mm DOI: 1.83 mm with 88.7% effciency 
Average Resolution at 10 mm DOI: 0.476 mm with 73.4% effciency 
Average Resolution at 15 mm DOI: 1.554 mm with 63.3% effciency 
Average Resolution at 20 mm DOI: 0.557 mm with 70.0% effciency 
Average Resolution at 25 mm DOI: 0.467 mm with 67.7% effcienc
```
