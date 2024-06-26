# SpatioTemporal-Lodging-
This is the coding implementation for the research project presented in the foloowing peer-review publication:

Varela, S.; Pederson, T.; Leakey, A. (2022). *Implementing Spatio-Temporal 3D-Convolution Neural Networks and UAV Time Series Imagery to Better Predict Lodging Damage in Sorghum*. Remote Sensing. [![DOI](https://img.shields.io/badge/DOI-10.3390/rs14030733-blue)](https://doi.org/10.3390/rs14030733) [![PDF](https://img.shields.io/badge/PDF-Download-orange)](papers/remotesensing-14-00733-v2.pdf)


1) check if anaconda is installed locally
```
conda --version

```
  https://docs.conda.io/projects/miniconda/en/latest/
  
  miniconda will be enough for creating venv and dependencies
  
  During the installation, you might be prompted to add Anaconda to the system PATH. If not, and if you encounter issues, you can add it manually:
  
  On Windows, you can check "Add Anaconda to my PATH environment variable" during installation.
  On Linux you may need to add the following line to your shell profile file (e.g., .bashrc or .zshrc):

Build local Conda virtual environment and dependencies
```
  conda create --name lodging python=3.9  

  conda activate lodging
  
  pip install -r requirements.txt
```
  
  tifffile==2023.2.28
  scikit-image==0.20.0
  numpy==1.24.2
  scikit-learn==1.0.1
  tensorflow==2.10.0
  pandas==1.5.3
  seaborn
  rasterio==1.3.6
  rasterstats==0.18.0

2) Execute lodging detection evaluation
```
  python lodging_detection.py --fit-type time-point 
```
  (for traditional time-point CNN analysis)

  or
  
  ```
  python lodging_detection.py --fit-type temporal 
```
  (for temporal integrated CNN analysis)

3) Execute lodging severity evaluation
```
  python lodging_severity.py --fit-type time-point
```
  (for traditional time-point CNN analysis)

  or
  ```
  python lodging_severity.py --fit-type temporal 
  ```
  (for temporal integrated CNN analysis)
