# Retinal Layer Segmentation and Analysis

This project uses the **NDD-SEG algorithm** to segment and analyze retinal layers in Optical Coherence Tomography (OCT) images. The code performs segmentation, boundary detection, thickness map generation, inpainting, and volumetric measurements for retinal layers.

---

## Features

1. **Segmentation with NDD-SEG**: Automatically segments retinal layers using the NDD-SEG deep learning model.
2. **Boundary Extraction**: Extracts layer boundaries from OCT images.
3. **Thickness Map Generation**: Creates thickness maps for different retinal layers.
4. **Inpainting**: Enhances thickness maps by filling irregularities.
5. **Volumetric Measurements**: Calculates the average thickness of retinal layers.
6. **Visualization**: Visualizes the B-scans, segmented boundaries, and thickness maps.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following Python libraries are installed:
   - `numpy`
   - `pandas`
   - `cv2` (OpenCV)
   - `matplotlib`
   - `scipy`

4. Import the **NDD-SEG**.

---

## File Structure

- **Input Folder**: Contains OCT images (`.png`) to be processed.
- **Output Folder**: Stores segmented boundaries, thickness maps, and volumetric measurements.
- **Pickle Files**: Save intermediate and final processed data.

---

## Usage

1. Place your OCT images in the specified `path1` directory.
2. Run the script:
   ```bash
   python <script_name>.py
   ```
3. The script performs the following steps:
   - Segments B-scans using the NDD-SEG model.
   - Saves segmented boundaries as `.pkl` files.
   - Visualizes B-scans and segmented boundaries.
   - Creates and saves thickness maps for retinal layers.
   - Calculates and exports volumetric measurements to an Excel file.

4. Check the outputs in the specified output directories:
   - **Segmented Boundaries**: `Boundaries_original_<file_name>.pkl`
   - **Thickness Maps**: Visualized as `.png` images.
   - **Volumetric Measurements**: Saved as Excel files.

---

## Example Results

### Visualizing Segmented Boundaries
B-scans overlaid with segmented boundaries:
- **RNFL**: Retinal Nerve Fiber Layer
- **GCIPL**: Ganglion Cell and Inner Plexiform Layer
- ... (list other layers)

### Thickness Maps
Visualized thickness maps for each layer:
- **RNFL**: RNFL/GCL interface - Inner Limiting Membrane
- ...

### Volumetric Measurements
An Excel file (`volumetric measurements original.xlsx`) summarizing thickness values for each retinal layer.

---

## Functions

### `inpaint(thicknessMap)`
Inpaints thickness maps to handle irregularities using Gaussian high-pass filtering.

### `inpaint_double(TM)`
Performs a second round of inpainting for smoothing.

---

## Requirements

- Python 3.7 or higher
- OCT images in `.png` format
- Proper installation of **NDD-SEG**

---

## Citation

If you use the NDD-SEG algorithm in your work, please cite:

> **A robust, flexible retinal segmentation algorithm designed to handle neuro-degenerative disease pathology (NDD-SEG)**

