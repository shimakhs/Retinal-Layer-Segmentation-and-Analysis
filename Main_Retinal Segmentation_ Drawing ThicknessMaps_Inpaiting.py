# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:53:17 2024

@author: Shima
"""

## Import libraries 
import pickle
import NDDSEG
import cv2
import numpy as np
from matplotlib import pyplot as plt, cm
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import pandas as pd


#####################################################################
### segment the data with NDD-SEG algorthm without any pre-processing
#####################################################################
path1 = '...'
dirs = os.listdir(path1)


# Filter and sort the filenames numerically
image_files = sorted([f for f in dirs if f.endswith(('.png'))], key=lambda x: int(os.path.splitext(x)[0]))


## import the NDD-SEG model
model = NDDSEG.DUNET(ds=2)

ThreeD_bscan = np.zeros((420, 750, 31))
X = np.zeros((31, 750, 10))
X1 = np.zeros((31, 750, 9))

dirData = os.listdir('.')
dirData = [filename for filename in dirData if filename.endswith('.pkl')]

# Creating a mask to calculate the Thicknesses of Retinal layers
mask = np.ones((512, 512))

for i in range(512):
    for j in range(512):
        if (i - 256) ** 2 + (j - 256) ** 2 > 256 ** 2:
            mask[i, j] = 0

# Opening each data             

for n, file_name in enumerate(image_files):
    full_file_name = os.path.join(path1, file_name)
    I = cv2.imread(full_file_name, 0)
    I = cv2.resize(I, (750, 420))


    ThreeD_bscan[:, :, n] = I

    # Apply the NDD-SEG model for segmentation
    J, P, T = model.predict(I)

    # segmented boundaries

    boundary = 0
    for k in J.keys():

        X[n, 0: len(J[k]['Y']), boundary] = J[k]['Y']
        boundary = boundary + 1
        X1[:,:,0:2] = X[:,:,0:2]
        X1[:,:,2:10] = X[:,:,3:11]
        
# save the boundaries of each zip file in a pickle
PicklePath = '...'
with open(PicklePath + '\\'+'Bscans_original_C001 BASELINE OS.pkl', 'wb') as f:
    pickle.dump(ThreeD_bscan, f)

PicklePath = '...'
with open(PicklePath + '\\'+'Boundaries_original_C001 BASELINE OS.pkl', 'wb') as f:
    pickle.dump(X1, f)        


############################################################
### show the original Bscans and their segmented boundaries
############################################################


# open Pickle files  

with open('...\\Bscans_original_C001 BASELINE OS' + '.pkl', 'rb') as f:
    bscans_original = pickle.load(f)
    
with open('...\\Boundaries_original_C001 BASELINE OS' + '.pkl', 'rb') as f:
    boundaries_original = pickle.load(f) 

# show the original Bscans and segmented boundaries on them 
    
for i in range(len(boundaries_original)):
    plt.figure()
    plt.imshow((bscans_original[:,:740,i]),cmap = 'gray')
    plt.plot(boundaries_original[i,:740,:], linewidth=3)  
    plt.grid(False)

# making the Thicknessmaps of original segmented data
all_thicknessmaps_original = []
for i in range(0,8):   
   Z =boundaries_original[:,:740,i+1] - boundaries_original[:,:740,i]
   Z = cv2.resize(Z,(512, 512))
   all_thicknessmaps_original.append(Z)          

# Plotting the ThicknessMaps of each of the Retinal Layers

plt.figure()
plt.imshow(all_thicknessmaps_original[0])
plt.title('RNFL : RNFL/GCL interface - Inner Limiting Membrane')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[1])
plt.title('GCIPL : IPL/INL interface - RNFL/GCL interface')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[2])
plt.title(' INL : INL/OPL interface - IPL/INL interface')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[3])
plt.title(' OPL : OPL/ONL interface - INL/OPL interface ')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[4])
plt.title(' ONL : External Limiting Membrane - OPL/ONL interface')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[5])
plt.title(' ELM : Inner boundary of EZ - External Limiting Membrane')
plt.colorbar()
plt.grid(False)

plt.figure()
plt.imshow(all_thicknessmaps_original[6])
plt.title(' EZ : Inner boundary of RPE/IZ complex - Inner boundary of EZ')
plt.colorbar()
plt.grid(False)


plt.figure()
plt.imshow(all_thicknessmaps_original[7])
plt.title("RPE : Bruch's Membrane - Inner boundary of RPE/IZ complex")
plt.colorbar()
plt.grid(False)


####### Calculating the volumetric measurements of each layer of the retina.
thick_allLayers = np.stack(all_thicknessmaps_original, axis=-1)

dirData = os.listdir('.')
dirData = [filename for filename in dirData if filename.endswith('.pkl')]


Final_results = pd.DataFrame(np.zeros((1,9)), columns=['Original_Image_Name', 'RNFL', 'GCIPL', 'INL', 'OPL', 'ONL', 'ELM', 'EZ', 'RPE'])

for layer in range(1,9):
     thickTemp = thick_allLayers[:, :, layer-1]
     mask2 = np.ones((512, 512))

     My_results = np.mean(np.mean(thickTemp[(mask == 1) & (mask2 == 1)]))


     #Final_results = Final_results.rename(index = {n : dirData[n]})
     Final_results.iloc[0,layer] = My_results
Final_results.to_excel(excel_writer =  '...\\volumetric measurements original.xlsx')



def inpaint(thicknessMap):
    """
    function to inpaint each thickness map before sending into algorithms
    """
    datathicknessMap = np.array(thicknessMap.astype('uint8'), dtype=float)

    # One way of making a highpass filter is to simply subtract a lowpass
    # filtered image from the original.
    lowpassthicknessMap = ndimage.gaussian_filter(datathicknessMap, 3) #sigma = 3 pixels
    gauss_highpassthicknessMap = datathicknessMap - lowpassthicknessMap
    thresholdedthicknessMap = gauss_highpassthicknessMap > 4
    thicknessMap_inpaint = cv2.inpaint(thicknessMap.astype('uint8'),thresholdedthicknessMap.astype('uint8'),3,cv2.INPAINT_TELEA)
    return gauss_highpassthicknessMap, thresholdedthicknessMap, thicknessMap_inpaint
    
def inpaint_double(TM):
    """
    function to replace big values (more than 3 times of the average) remained after first inpainting and puts the average value instesd. then another inpainting is performed to smooth up the uneven parts  
    """
    a, Th_TM, TM_inpaint = inpaint(TM)
    TM_inpaint[TM_inpaint > (3 * np.mean(TM_inpaint[Th_TM==0]))] = np.mean(TM_inpaint[Th_TM==0])
    a, b, TM_inpaint2 = inpaint(TM_inpaint)
    return TM_inpaint2
    
    
    
# making the Thicknessmaps after inpaiting 
all_thicknessmaps_inpaited_original = []
for i in range(0,8):   
   Z =boundaries_original[:,:740,i+1] - boundaries_original[:,:740,i]
   gauss_highpassZ, thresholdedZ, Z_inpaint = inpaint(Z)  
   Z_inpaint2 = inpaint_double(Z_inpaint)
   #z = np.fliplr(Z_inpaint2)
   #z = cv2.resize(z,(256, 256))  
   z = cv2.resize(Z_inpaint2,(512, 512))   

   all_thicknessmaps_inpaited_original.append(z)          
     
    



####### Calculating the volumetric measurements of each layer of the retina after Inpaiting
thick_allLayers_original_inpainted = np.stack(all_thicknessmaps_inpaited_original, axis=-1)

mask = np.ones((512, 512))

for i in range(512):
    for j in range(512):
        if (i - 256) ** 2 + (j - 256) ** 2 > 256 ** 2:
            mask[i, j] = 0



dirData = os.listdir('.')
dirData = [filename for filename in dirData if filename.endswith('.pkl')]
#counter = len(dirData)


Final_results_original_inpainted = pd.DataFrame(np.zeros((1,9)), columns=['Original_Image_Name', 'RNFL', 'GCIPL', 'INL', 'OPL', 'ONL', 'ELM', 'EZ', 'RPE'])

for layer in range(1,9):
     thickTemp = thick_allLayers_original_inpainted[:, :, layer-1]
     mask2 = np.ones((512, 512))
#C[count][layer] = np.mean(np.mean(thickTemp[(mask == 1) & (mask2 == 1)]))

     My_results = np.mean(np.mean(thickTemp[(mask == 1) & (mask2 == 1)]))


     #Final_results = Final_results.rename(index = {n : dirData[n]})
     Final_results_original_inpainted.iloc[0,layer] = My_results
Final_results_original_inpainted.to_excel(excel_writer =  '...\\volumetric measurements original inpainted.xlsx')




##### Draw Thickness map of Whole Retina ######

whole_retina = []

Z =boundaries_original[:,:740,8] - boundaries_original[:,:740,0]
gauss_highpassZ, thresholdedZ, Z_inpaint = inpaint(Z)  
Z_inpaint2 = inpaint_double(Z_inpaint)
   #z = np.fliplr(Z_inpaint2)
   #z = cv2.resize(z,(256, 256))  
z = cv2.resize(Z_inpaint2,(512, 512))   

whole_retina.append(z)          
     
    
for n in range(len(whole_retina)):
    plt.figure()
    plt.imshow(whole_retina[n])
    plt.title('Thicknessmap of whole retina')
    plt.colorbar()
    plt.grid(False)
    


    
    

