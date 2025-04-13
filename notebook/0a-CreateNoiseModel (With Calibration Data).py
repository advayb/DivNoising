#!/usr/bin/env python
# coding: utf-8

# # Generate a Noise Model using Calibration Data 
# 
# We will use pairs of noisy calibration observations $x_i$ and clean signal $s_i$ (created by averaging these noisy, calibration images) to estimate the conditional distribution $p(x_i|s_i)$. Histogram-based and Gaussian Mixture Model-based noise models are generated and saved. 
# 
# __Note:__ Noise model can also be generated if calibration data is not available. In such a case, we use an approach called ```Bootstrapping```. Take a look at the notebook ```0b-CreateNoiseModel (With Bootstrapping)``` on how to do so. To understand more about the ```Bootstrapping``` procedure, take a look at the readme [here](https://github.com/juglab/PPN2V).

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import torch
import os
import urllib
import zipfile
from torch.distributions import normal
import matplotlib.pyplot as plt, numpy as np, pickle
from scipy.stats import norm
from tifffile import imread, imwrite as imsave
import sys

# Determine if running in Colab
IN_COLAB = 'google.colab' in sys.modules

# If in Colab, set up paths properly
if IN_COLAB:
    # Ensure the model module is in the path
    if not os.path.exists('model'):
        # Clone the repository if needed
        import subprocess
        subprocess.run(["git", "clone", "https://github.com/yourusername/DivNoising-VAE.git"])
        os.chdir("DivNoising-VAE")
    
    # Add the necessary paths
    sys.path.append('./')
    from model.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
    import model.histNoiseModel as histNoiseModel
    from model.utils import plotProbabilityDistribution
else:
    # Local machine paths
    # Add the parent directory and model directory to the path
    sys.path.append('../')
    sys.path.append('../model')
    # Import from the local model directory instead of divnoising module
    from model.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
    import model.histNoiseModel as histNoiseModel
    from model.utils import plotProbabilityDistribution

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


# ### Download data
# 
# Download the data from https://zenodo.org/record/5156960/files/Mouse%20skull%20nuclei.zip?download=1. Here we show the pipeline for Mouse nuclei dataset. Save the dataset in an appropriate path. For us, the path is the data folder which exists at `./data`.

# In[2]:


# Check if in Colab and handle data accordingly
if IN_COLAB:
    from google.colab import files
    
    # Create data directory if it doesn't exist
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    
    # Check if data is already present
    if not os.path.exists('./data/Mouse_skull_nuclei/edgeoftheslide_300offset.tif'):
        # Option 1: Manual upload
        print("Please upload the Mouse_skull_nuclei.zip file:")
        uploaded = files.upload()  # This will open a file picker
        
        # Extract the uploaded zip
        zip_filename = next(iter(uploaded.keys()))
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('./data/')
        
        # Option 2: Direct download (uncomment to use this instead)
        # zipPath="./data/Mouse_skull_nuclei.zip"
        # if not os.path.exists(zipPath):  
        #     urllib.request.urlretrieve('https://zenodo.org/record/5156960/files/Mouse%20skull%20nuclei.zip?download=1', zipPath)
        #     with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        #         zip_ref.extractall("./data")
else:
    # Original download code for local machine
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    
    zipPath="./data/Mouse_skull_nuclei.zip"
    if not os.path.exists(zipPath):  
        data = urllib.request.urlretrieve('https://zenodo.org/record/5156960/files/Mouse%20skull%20nuclei.zip?download=1', zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("./data")


# The noise model is a characteristic of your camera and not of the sample. The downloaded data folder contains a set of calibration images (For the Mouse nuclei dataset, it is ```edgeoftheslide_300offset.tif``` showing the edge of a slide and the data to be denoised is named ```example2_digital_offset300.tif```). The calibration images can be anything which is static and imaged multiple times in succession. Thus, the edge of slide works as well. We can either bin the noisy - GT pairs (obtained from noisy calibration images) as a 2-D histogram or fit a GMM distribution to obtain a smooth, parametric description of the noise model.

# Specify ```path``` where the noisy calibration data will be loaded from. It is the same path where noise model will be stored when created later, ```dataName``` is the name you wish to have for the noise model,  ```n_gaussian``` to indicate how many Gaussians willbe used for learning a GMM based noise model, ```n_coeff``` for indicating number of polynomial coefficients will be used to patrametrize the mean, standard deviation and weight of GMM noise model. The default settings for ```n_gaussian``` and ```n_coeff``` generally work well for most datasets.

# In[ ]:


path="./data/Mouse_skull_nuclei/"
# Check if file exists before trying to load it
if not os.path.exists(path+'edgeoftheslide_300offset.tif'):
    raise FileNotFoundError(f"Could not find data file at {path}edgeoftheslide_300offset.tif. Please make sure you've uploaded the data correctly.")

observation= imread(path+'edgeoftheslide_300offset.tif') # Load the appropriate calibration data

dataName = 'nuclei' # Name of the noise model 
n_gaussian = 3 # Number of gaussians to use for Gaussian Mixture Model
n_coeff = 2 # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.


# In[ ]:


nameHistNoiseModel ='HistNoiseModel_'+dataName+'_'+'calibration'
nameGMMNoiseModel = 'GMMNoiseModel_'+dataName+'_'+str(n_gaussian)+'_'+str(n_coeff)+'_'+'calibration'


# In[ ]:


# The data contains 100 images of a static sample (edge of a slide).
# We estimate the clean signal by averaging all images.

signal=np.mean(observation[:, ...],axis=0)[np.newaxis,...]

# Let's look the raw data and our pseudo ground truth signal
print(signal.shape)
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label='average (ground truth)')
plt.imshow(signal[0],cmap='gray')
plt.subplot(1, 2, 1)
plt.title(label='single raw image')
plt.imshow(observation[0],cmap='gray')
plt.show()


# ### Creating the Histogram Noise Model

# Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a histogram based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$. 

# In[ ]:


# We set the range of values we want to cover with our model.
# The pixel intensities in the images you want to denoise have to lie within this range.
minVal, maxVal = 2000, 22000
bins = 400

# We are creating the histogram.
# This can take a minute.
histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, observation,signal)

# Saving histogram to disc.
np.save(path+nameHistNoiseModel+'.npy', histogram)
histogramFD=histogram[0]


# In[ ]:


# Let's look at the histogram-based noise model.
plt.xlabel('Observation Bin')
plt.ylabel('Signal Bin')
plt.imshow(histogramFD**0.25, cmap='gray')
plt.show()


# ### Creating the GMM noise model
# Using the raw pixels $x_i$, and our averaged GT $s_i$, we are now learning a GMM based noise model. It describes the distribution $p(x_i|s_i)$ for each $s_i$. 

# In[ ]:


min_signal=np.min(signal)
max_signal=np.max(signal)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)


# Iterating the noise model training for `n_epoch=4000` and `batchSize=25000` works the best for `Mouse nuclei` dataset. 

# In[ ]:


gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = min_signal, max_signal =max_signal, 
                                                      path=path, weight = None, n_gaussian = n_gaussian, 
                                                      n_coeff = n_coeff, min_sigma = 50, device = device)


# In[ ]:


gaussianMixtureNoiseModel.train(signal, observation, batchSize = 25000, n_epochs = 4000, 
                                learning_rate=0.1, name = nameGMMNoiseModel)


# ### Visualizing the Histogram-based and GMM-based noise models

# In[ ]:


plotProbabilityDistribution(signalBinIndex=170, histogram=histogramFD, 
                            gaussianMixtureNoiseModel=gaussianMixtureNoiseModel, min_signal=minVal, 
                            max_signal=maxVal, n_bin= bins, device=device)


# In[ ]:




