#!/usr/bin/env python
# coding: utf-8

# # DivNoising - Prediction
# This notebook contains an example on how to use a previously trained DivNoising VAE to denoise images.
# If you haven't done so please first run ```0-CreateNoiseModel.ipynb``` and ```1-Training.ipynb```, which will download the data, create a noise model and train the DivNoising model.  

# In[ ]:


# We import all our dependencies.
import numpy as np
import torch
import time
import sys
import os

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
    import os
    from tifffile import imread, imwrite as imsave
    # Import from model directory
    from model import utils
    from nets import lightningmodel
    from glob import glob
    from matplotlib import pyplot as plt
else:
    # Local machine paths
    # Add the parent directory and model directory to the path
    sys.path.append('../')
    sys.path.append('../model')
    # Import from local model directory instead of divnoising
    from model import utils
    from nets import lightningmodel
    from glob import glob
    from tifffile import imread, imwrite as imsave
    from matplotlib import pyplot as plt

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
if not torch.cuda.is_available():
    print("Warning: GPU not found, predictions will run on CPU and can be somewhat slow!")


# # Load data to predict on
# The data should be present in the directory specified by ```noisy_data_path``` and the ```noisy_input``` is the name of the image in this directory that needs to be denoised. 
# This notebook expects 2D datasets in ```.tif``` format. If your data is a stack of 2D images, you can load it as shown in the next cell. If you dataset has multiple individual 2D tif files, comment out the second line in the cell below and uncomment the third line.

# In[ ]:


# Check if in Colab and handle data accordingly
if IN_COLAB:
    from google.colab import files
    
    # Create data directory if it doesn't exist
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./data/Mouse_skull_nuclei'):
        os.mkdir('./data/Mouse_skull_nuclei')
    
    # Check if data is already present
    if not os.path.exists('./data/Mouse_skull_nuclei/example2_digital_offset300.tif'):
        # Option 1: Manual upload
        print("Please upload the dataset file (example2_digital_offset300.tif):")
        uploaded = files.upload()  # This will open a file picker
        
        # Move the uploaded file to the correct location
        for filename in uploaded.keys():
            os.rename(filename, f"./data/Mouse_skull_nuclei/{filename}")

noisy_data_path="./data/Mouse_skull_nuclei/"
# Check if file exists before trying to load it
if not os.path.exists(noisy_data_path+'example2_digital_offset300.tif'):
    raise FileNotFoundError(f"Could not find data file at {noisy_data_path}example2_digital_offset300.tif. Please make sure you've uploaded the data correctly.")

noisy_input = imread(noisy_data_path+'example2_digital_offset300.tif').astype(np.float32)
# noisy_input= imread(noisy_data_path+'*.tif').astype(np.float32) # To load multiple individual 2D tif images


# # Load our model
# We load the last weights of the trained model from the ```basedir```. The ```basedir``` should be the same which you specified in the training notebook `1-Training.ipynb`. Also specify the ```model_name```. It should be the same as specified in the training notebook `1-Training.ipynb`. 

# In[ ]:


# In Colab, create directories and upload model if needed
if IN_COLAB:
    # Create models directory if it doesn't exist
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    
    # Check if model is already present
    if not glob('./models/divnoising_mouse_skull_nuclei_demo_last.ckpt'):
        print("Please upload your trained model checkpoint (divnoising_mouse_skull_nuclei_demo_last.ckpt):")
        uploaded = files.upload()  # This will open a file picker
        
        # Move the uploaded file to the correct location
        for filename in uploaded.keys():
            os.rename(filename, f"./models/{filename}")

basedir = 'models'
model_name = 'divnoising_mouse_skull_nuclei_demo'

# Find the model checkpoint file
model_files = glob(f"{basedir}/{model_name}_last.ckpt")
if not model_files:
    raise FileNotFoundError(f"Could not find model file at {basedir}/{model_name}_last.ckpt. Please make sure you've uploaded the model correctly.")

name = model_files[0]
vae = lightningmodel.VAELightning.load_from_checkpoint(checkpoint_path = name)
vae.to(device)


# # Here we predict some qualitative diverse solutions

# In[ ]:


utils.plot_qualitative_results(noisy_input, vae, device)


# # Predict denoised images and (optionally) save them
# Specify how many denoised samples need to be predicted for each noisy image by specifying the parameter ```num_samples```. Also MMSE denoised estimate will be computed using these many samples.  
# 
# If you do not want access to different samples but just need the MMSE denoised estimate, set the paarmeter ```returnSamples=False```.
# 
# You can also save the denoised results (both samples and MMSE estimate for each noisy image) by providing the ```export_results_path``` which is the directory where the results should be saved.
# 
# Alternatively, you can also export the MMSE estimate and only a fraction of the samples used for computing MMSE estimate for each image by specifying the parameter ```fraction_sample_to_export```. If set to $0$, none of the samples are exported and only the MMSE estimate is exported, whereas setting it to $1$ exports all samples used for computing MMSE estimate.
# 
# If you only want to export MMSE estimate, set parameter ```export_mmse``` to True. If you do not want to export MMSE estimate, set it to ```False```.
# 
# The parameter ```tta``` refers to test time augmentation which may improve performance of DivNoising even further but will take ```8x``` longer to predict. This is enabled by default. If you wish to disable it, set it to ```False```.

# In[ ]:


num_samples = 1000
export_results_path = "denoised_results"
fraction_samples_to_export = 0
export_mmse = False
tta = True
mmse_results = utils.predict_and_save(noisy_input,vae,num_samples,device,
                                fraction_samples_to_export,export_mmse,export_results_path,tta)


# # Compute PSNR
# Here we compute Peak Signal-to-Noise Ratio (PSNR) of the denoised MMSE output with respect to the available GT data specified by the ```gt``` parameter in the next cell. If you do not have GT data, do not run this cell.

# In[ ]:


PSNRs=[]
gt=np.mean(noisy_input[:,...],axis=0)[np.newaxis,...]

for i in range(len(mmse_results)):
    psnr=utils.PSNR(gt[0],mmse_results[i])
    PSNRs.append(psnr)
    print("image:", i, "psnr:"+format(psnr,".3f")+ "\t mean psnr:"+format(np.mean(PSNRs),".3f")) 
    time.sleep(0.5)
    
print('mean',np.mean(PSNRs))


# In[ ]:




