# -*- coding: utf-8 -*- 
"""
Created on Fri Nov 22 11:53:17 2024

Author: Shima

Description:
This script processes OCT images to segment retinal layers using the NDD-SEG algorithm, 
computes thickness maps, and generates volumetric measurements for each retinal layer.

Inputs:
- OCT images stored as `.png` files in a specified directory.
- Pretrained NDD-SEG model for segmentation.

Outputs:
- Pickle files storing the segmented boundaries and original B-scans.
- Excel files containing volumetric measurements of retinal layers.
- Thickness maps for individual layers and the whole retina, displayed and saved.

"""

# Import libraries
import os
import cv2
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import ndimage
import NDDSEG  # Assuming a custom library for NDD-SEG model

# Helper Functions
def create_circular_mask(diameter):
    """
    Create a circular mask of specified diameter.
    
    Parameters:
    - diameter (int): Diameter of the circular mask.
    
    Returns:
    - mask (ndarray): A binary mask with a circular region set to 1.
    """
    mask = np.ones((diameter, diameter))
    center = diameter // 2
    radius = center
    for i in range(diameter):
        for j in range(diameter):
            if (i - center) ** 2 + (j - center) ** 2 > radius ** 2:
                mask[i, j] = 0
    return mask

def save_pickle(data, file_path):
    """
    Save data to a pickle file.
    
    Parameters:
    - data (object): Data to be saved.
    - file_path (str): Destination file path for the pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    """
    Load data from a pickle file.
    
    Parameters:
    - file_path (str): Path to the pickle file.
    
    Returns:
    - data (object): Loaded data from the pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def inpaint_thickness_map(thickness_map):
    """
    Perform inpainting on a thickness map to correct invalid regions.
    
    Parameters:
    - thickness_map (ndarray): Input thickness map.
    
    Returns:
    - inpainted_map (ndarray): Corrected thickness map.
    """
    thickness_map = thickness_map.astype('uint8')
    lowpass_map = ndimage.gaussian_filter(thickness_map, 3)  # Apply Gaussian smoothing
    highpass_map = thickness_map - lowpass_map
    thresholded_map = highpass_map > 4  # Thresholding
    inpainted_map = cv2.inpaint(thickness_map, thresholded_map.astype('uint8'), 3, cv2.INPAINT_TELEA)
    return inpainted_map

def calculate_layer_thickness(boundaries, layer_index):
    """
    Calculate the thickness of a specific retinal layer from boundaries.
    
    Parameters:
    - boundaries (ndarray): Array containing boundary data.
    - layer_index (int): Index of the layer to calculate thickness for.
    
    Returns:
    - thickness_map (ndarray): Thickness map for the specified layer.
    """
    return boundaries[:, :, layer_index + 1] - boundaries[:, :, layer_index]

# Main Script
def main():
    # Paths
    input_dir = '...'  # Replace with your image directory
    pickle_output_dir = '...'  # Replace with the directory to save pickle files
    excel_output_path = '...\\volumetric_measurements.xlsx'
    
    # Parameters
    image_shape = (750, 420)  # Desired shape for resizing images
    num_layers = 8  # Number of retinal layers
    
    # Load and preprocess images
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')], 
                         key=lambda x: int(os.path.splitext(x)[0]))
    three_d_bscans = np.zeros((420, 750, len(image_files)))
    
    for n, file_name in enumerate(image_files):
        full_path = os.path.join(input_dir, file_name)
        image = cv2.imread(full_path, 0)  # Load in grayscale
        image = cv2.resize(image, image_shape)
        three_d_bscans[:, :, n] = image
    
    # Segment layers using the NDD-SEG model
    model = NDDSEG.DUNET(ds=2)
    segmented_boundaries = np.zeros((len(image_files), image_shape[1], num_layers))
    
    for n in range(len(image_files)):
        segmented_data, _, _ = model.predict(three_d_bscans[:, :, n])
        for idx, key in enumerate(segmented_data.keys()):
            segmented_boundaries[n, :, idx] = segmented_data[key]['Y']
    
    # Save B-scans and boundaries
    save_pickle(three_d_bscans, os.path.join(pickle_output_dir, 'Bscans.pkl'))
    save_pickle(segmented_boundaries, os.path.join(pickle_output_dir, 'Boundaries.pkl'))
    
    # Calculate thickness maps
    thickness_maps = []
    for layer_index in range(num_layers):
        thickness_map = calculate_layer_thickness(segmented_boundaries, layer_index)
        thickness_map_resized = cv2.resize(thickness_map, (512, 512))
        thickness_maps.append(thickness_map_resized)
    
    # Visualize and save results
    plt.figure()
    for idx, map_data in enumerate(thickness_maps):
        plt.imshow(map_data, cmap='viridis')
        plt.title(f'Layer {idx + 1} Thickness Map')
        plt.colorbar()
        plt.show()
    
    # Compute volumetric measurements
    circular_mask = create_circular_mask(512)
    results = pd.DataFrame(columns=['Layer', 'Mean Thickness'])
    
    for idx, map_data in enumerate(thickness_maps):
        masked_map = map_data[circular_mask == 1]
        mean_thickness = np.mean(masked_map)
        results = pd.concat([results, pd.DataFrame({'Layer': [idx + 1], 'Mean Thickness': [mean_thickness]})])
    
    results.to_excel(excel_output_path, index=False)
    print(f"Volumetric measurements saved to {excel_output_path}")

if __name__ == "__main__":
    main()
