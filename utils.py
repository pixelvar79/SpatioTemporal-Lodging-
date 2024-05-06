""" Helper script to load image (x) and target (y) dataset """
# Import libraries
#from PIL import Image
from tifffile import imread, imwrite
from skimage.transform import resize
import numpy as np
import pandas as pd
from pathlib import Path

# Functions to load dataset
def load_image(picture):
    """
    load tiff files from directory
    """
    img = imread(picture)
    img = resize(img,(48,48,55))
    return(img)


def load_dataset(img_dir, gt_dir, task='classification'):
    """
    Loads x and y dataset for classification or regression
    :param data_dir: path to data folder containing images and CSV file
    :param task: 'classification' or 'regression', default is 'classification'
    :return: x and y dataset
    """
    # loads x data, images sorted according to plot number
    img_list = [load_image(file) for file in sorted(Path(img_dir).glob('*.tif'),
                                                    key=lambda x: int(x.stem.split('_')[1]))]
    x = np.stack(img_list)
    # loads y data
    csv_file = Path(gt_dir)/'lodging_20191.csv'
    df = pd.read_csv(csv_file)
    if task == 'regression':
        df = df[df['Lodging'] == 'yes']  # select rows for only biomass values
        y = df['Score']
        x = [x[i] for i in df.index]  # slice x for only avaialbe biomass values
        x = np.array(x)
    else:
        y = df['Lodging']
    return x,y