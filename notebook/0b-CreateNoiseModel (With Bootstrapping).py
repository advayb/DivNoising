#!/usr/bin/env python
# coding: utf-8

# # Generate a Noise Model using Bootstrapping
# 
# Here we assume that we do not have access to calibration data to create a noise model for training DivNoising. In this case, we use an approach called ```Bootstrapping``` to create a noise model from noisy data itself. The idea is that we will first use the unsupervised denoising method Noise2Void to denoise our data, and use this denoised image as a pseudo ground truth to create a noise model. The noise model created through this approach is not perfect, but since this is only used for training DivNoising, results are still pretty good. 
# 
# DivNoising when using bootstrapped noise model generally gives better results compared to Noise2Void denoising. Also, unlike Noise2Void, we additionally obtain diverse denoised samples corresponding to the noisy input which better represent the inherent uncertainty in the solution.
# 
# __Note:__ Denoising methods other than Noise2Void can also be used to obtain pseudo GT for bootsrapping a noise model.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import torch
import os
import urllib
import zipfile
from torch.distributions import normal
import numpy as np
import pickle
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
    # Import from local model directory instead of divnoising
    from model.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
    import model.histNoiseModel as histNoiseModel
    from model.utils import plotProbabilityDistribution

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


# ### Download data
# 
# Download the data from https://zenodo.org/record/5156960/files/Mouse%20skull%20nuclei.zip?download=1. Here we show the pipeline for Convallaria dataset. Save the dataset in an appropriate path. For us, the path is the data folder which exists at `./data`.

# In[ ]:


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


# In[ ]:

path = './data/Mouse_skull_nuclei/'
# Check if file exists before trying to load it
if not os.path.exists(path+'edgeoftheslide_300offset.tif'):
    raise FileNotFoundError(f"Could not find data file at {path}edgeoftheslide_300offset.tif. Please make sure you've uploaded the data correctly.")

observation = imread(path+'edgeoftheslide_300offset.tif') #Load the noisy data to be denoised


# ### Load pseudo GT
# 
# As described above, we will use the denoising results obtained by Noise2Void and treat them as pseudo GT corresponding to our noisy data. Following this, we will use the pair of noisy images and corresponding Noise2Void denoised images to learn a noise model. You can use any other denoising method as well and treat their denoised result as pseudo GT to learn a noise model for DivNoising training.
# 
# If you have access to pseudo GT (denoised images from some other denoising method), provide the directory path for these images in ```pseudo_gt_path``` parameter. If you do not have such pseudo GT, first generate these images by running any denoising method on your data. For example, you can use Noise2Void denoising as shown [here](https://github.com/juglab/n2v).
# 
# Next, specify the directory path (```noisy_input_path```) to the noisy data that you wish to denoise with DivNoising.
# 
# Using these, we can either bin the noisy - pseudo GT pairs as a 2-D histogram or fit a GMM distribution to obtain a smooth, parametric description of the noise model.

# In[ ]:


# In Colab, we might need to create a pseudo_gt directory or upload files
if IN_COLAB and not os.path.exists('./pseudo_gt'):
    os.mkdir('./pseudo_gt')
    print("Please upload your pseudo GT denoised images (*.tif files):")
    files.upload()  # This will open a file picker
    
    # Move uploaded files to pseudo_gt directory
    import glob
    for f in glob.glob("*.tif"):
        os.rename(f, f"./pseudo_gt/{f}")

pseudo_gt_path="./pseudo_gt/"
# Check for files in the directory
if not os.listdir(pseudo_gt_path):
    print(f"No files found in {pseudo_gt_path}. For this example, we'll use the observation as pseudo GT.")
    # As a fallback, use the observation data as pseudo GT (for demonstration only)
    if not os.path.exists(pseudo_gt_path):
        os.makedirs(pseudo_gt_path)
    imwrite(f"{pseudo_gt_path}/pseudo_gt.tif", observation[0])
    signal = imread(f"{pseudo_gt_path}/pseudo_gt.tif")[np.newaxis, ...]
else:
    signal = imread(pseudo_gt_path+'*.tif') # Load pseudo GT (obtained as a result of denoising from other methods)


# Continue with the rest of the code...

# Specify ```path``` where the noisy data will be loaded from. It is the same path where noise model will be stored when created later, ```dataName``` is the name you wish to have for the noise model,  ```n_gaussian``` to indicate how mamny Gaussians willbe used for learning a GMM based noise model, ```n_coeff``` for indicating number of polynomial coefficients will be used to patrametrize the mean, standard deviation and weight of GMM noise model. The default settings for ```n_gaussian``` and ```n_coeff``` generally work well for most datasets.

# In[ ]:


path = './data/Mouse_skull_nuclei/'
dataName = 'mouse_skull_nuclei' # Name of the noise model 
n_gaussian = 3 # Number of gaussians to use for Gaussian Mixture Model
n_coeff = 2 # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.


# In[ ]:


nameHistNoiseModel ='HistNoiseModel_'+dataName+'_'+'bootstrap'
nameGMMNoiseModel = 'GMMNoiseModel_'+dataName+'_'+str(n_gaussian)+'_'+str(n_coeff)+'_'+'bootstrap'


# In[ ]:


# Let's look the raw data and our pseudo ground truth signal
print(signal.shape)
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label='pseudo ground truth')
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


min_signal=np.percentile(signal, 0.5)
max_signal=np.percentile(signal, 99.5)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)


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


gaussianMixtureNoiseModel.train(signal, observation, batchSize = 25000, n_epochs = 4000, learning_rate=0.1, 
                                name = nameGMMNoiseModel, lowerClip = 0.5, upperClip = 99.5)


# ### Visualizing the Histogram-based and GMM-based noise models

# In[ ]:


plotProbabilityDistribution(signalBinIndex=170, histogram=histogramFD, 
                            gaussianMixtureNoiseModel=gaussianMixtureNoiseModel, min_signal=minVal, 
                            max_signal=maxVal, n_bin= bins, device=device)


# In[ ]:




