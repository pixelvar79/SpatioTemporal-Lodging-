from tifffile import imread, imwrite
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt, operator as op
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential 
from tensorflow.keras.backend import set_image_data_format 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization,GlobalAveragePooling3D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras import optimizers, losses, utils 
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score,roc_curve, auc,roc_auc_score,f1_score,jaccard_score,precision_score,recall_score
from PIL import Image
import gc
from time import sleep