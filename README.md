# SpatioTemporal-Lodging-
This is the coding implementation for the research project presented in the foloowing peer-review publication:

Varela, S.; Pederson, T.; Leakey, A. (2022). *Implementing Spatio-Temporal 3D-Convolution Neural Networks and UAV Time Series Imagery to Better Predict Lodging Damage in Sorghum*. Remote Sensing. [![DOI](https://img.shields.io/badge/DOI-10.3390/rs14030733-blue)](https://doi.org/10.3390/rs14030733) [![PDF](https://img.shields.io/badge/PDF-Download-orange)](papers/remotesensing-14-00733-v2.pdf)


1) Build local Conda virtual environment and install dependencies

  conda create --name lodging python=3.9  
  
  cd lodging
  
  conda activate lodging
  
  pip install -r requirements.txt

2) Execute lodging_detection.py for lodging detection

  python lodging_detectio.py 
  


3) Execute loding_severity.py for lodging severity evaluation

   python lodging_severity.py 

