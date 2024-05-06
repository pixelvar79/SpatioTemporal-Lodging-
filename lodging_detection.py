import utils
from utils import load_dataset

from tifffile import imread, imwrite
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, f1_score, jaccard_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import gc
from time import sleep

# Load x, y dataset
X, y = utils.load_dataset(dir_img, dir_gt, task='classification')

print(X.shape, y.shape)

# Split x, y dataset
RANDOM_STATE = 123

def data_splitting(x_var, y_var):
    encoder = LabelEncoder()
    encoder.fit(y_var)
    y_var = encoder.transform(y_var)

    global x_train, x_test, y_train, y_test, x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, train_size=0.7, random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=RANDOM_STATE)
    sample_shape = x_train[0].shape

def fit_model():
    global model
    # Model configuration
    MODEL_NAME = f'{name}.h5'
    SAMPLE_SHAPE = x_train[0].shape
    EPOCHS = 100
    SIZE = 32
    MODE = 'max'
    METRIC_VAR = 'val_accuracy'
    LR = 0.01
    DC = 0.001
    OPT = tf.keras.optimizers.Adam(learning_rate=LR, decay=DC)
    es = EarlyStopping(monitor=METRIC_VAR, verbose=1, mode=MODE, min_delta=0.01, patience=10)
    mc = ModelCheckpoint(MODEL_NAME, monitor=METRIC_VAR, mode=MODE, save_best_only=True, verbose=1)
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=SAMPLE_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])

def model_performance():
    global saved_model
    saved_model = tf.keras.models.load_model(MODEL_NAME)
    score = model.evaluate(x_test, y_test)
    score1 = saved_model.evaluate(x_test, y_test)
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='upper right')  
    plt.show()

def results_df():
    global cnn_flower, pred
    pred = (model.predict(x_test) > 0.5).astype("int32")
    cnn_flower = pd.DataFrame()   
    cnn_flower['obs'] = y_test
    cnn_flower['pred'] = pred
    cnn_flower.to_csv(f'{name}_.csv', sep='\t')

def roc_plot_results():
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = saved_model.predict(x_test)
    cnn_prob = pd.DataFrame()   
    cnn_prob['random'] = ns_probs
    cnn_prob['obs'] = y_test
    cnn_prob['pred'] = lr_probs
    cnn_prob.to_csv(f'{name}_prob_.csv', sep='\t')
    ns_auc = roc_auc_score(cnn_prob['obs'], cnn_prob['random'])
    lr_auc = roc_auc_score(cnn_prob['obs'], cnn_prob['pred'])
    ns_fpr, ns_tpr, _ = roc_curve(cnn_prob['obs'], cnn_prob['random'])
    lr_fpr, lr_tpr, _ = roc_curve(cnn_prob['obs'], cnn_prob['pred'])
    plt.plot(ns_fpr, ns_tpr, linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def summary_metrics():
    accur = []
    roc = []
    f1 = []
    jaccard = []
    precision = []
    recall = []
    m1 = accuracy_score(cnn_flower['obs'], cnn_flower['pred'])
    m2 = roc_auc_score(cnn_flower['obs'], cnn_flower['pred'])
    m3 = f1_score(cnn_flower['obs'], cnn_flower['pred'])
    m4 = jaccard_score(cnn_flower['obs'], cnn_flower['pred'], average='micro')
    m5 = precision_score(cnn_flower['obs'], cnn_flower['pred'])
    m6 = recall_score(cnn_flower['obs'], cnn_flower['pred'])
    accur.append(np.asarray(m1))
    roc.append(np.asarray(m2))
    f1.append(np.asarray(m3))
    jaccard.append(np.asarray(m4))
    precision.append(np.asarray(m5))
    recall.append(np.asarray(m6))
    mdf = pd.DataFrame()
    mdf['accuracy'] = pd.DataFrame(accur)
    mdf['roc_score'] = pd.DataFrame(roc)
    mdf['f1'] = pd.DataFrame(f1)
    mdf['jaccard'] = pd.DataFrame(jaccard)
    mdf['precision'] = pd.DataFrame(precision)
    mdf['recall'] = pd.DataFrame(recall)
    mdf['label'] = pd.DataFrame({name})
    mdf['model'] = 'MS_TP_C'
    dfs.append(mdf)

slices = (('75', slice(35, 40)),
          ('83', slice(40, 45)),
          ('88', slice(45, 50)),
          ('97', slice(50, 55)))

list_data = []
labels = []

for names, slicing in slices:
    list_data.append(X[:, :, :, slicing])
    labels.append(names)

dfs = [] 
list_fpr = []
list_tpr = []
time_list = []
memory_list = []

for i in range(1):
    for name, dates in zip(labels, list_data):
        data_splitting(dates, y)
        fit_model()
        model_performance()
        results_df()
        roc_plot_results()
        summary_metrics()

