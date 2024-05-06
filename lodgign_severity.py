import utils
from utils import load_dataset
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt

import argparse
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dropout, Flatten, Dense, GlobalAveragePooling3D


# Load x, y dataset from local directory
X, y = utils.load_dataset(dir_img, dir_gt, task='regression')

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

#time-point 2D-CNN model classifier
def fit_model():
    MODEL_NAME = f'{name}.h5'
    SAMPLE_SHAPE =  x_train[0].shape
    EPOCHS = 100
    SIZE = 16
    MODE = 'min'
    METRIC_VAR = 'val_loss'
    LR = 0.001
    DC = 0.0001
    OPT = tf.keras.optimizers.Adam(learning_rate=LR, decay=DC)
    #es = callbacks.EarlyStopping(monitor=METRIC_VAR,verbose=1,mode=MODE,min_delta=0.01,patience=50);
    #mc = callbacks.ModelCheckpoint(MODEL_NAME, monitor=METRIC_VAR, mode=MODE, save_best_only=True,verbose=1);
    global model, history
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=SAMPLE_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',optimizer=OPT,metrics=['mae']);
    # Train model
    history = model.fit(x_train, y_train,epochs=EPOCHS,batch_size=SIZE,validation_data=(x_val, y_val),verbose=2)
    
    
#time-point 3D-CNN model classifier
def fit_model1():
    # Model configuration
    global model, history
    MODEL_NAME = f'{name}.h5'
    SAMPLE_SHAPE =  x_train[0].shape
    EPOCHS = 100
    SIZE = 16
    MODE = 'min'
    METRIC_VAR = 'val_loss'
    LR = 0.001
    DC = 0.0001
    OPT = tf.keras.optimizers.Adam(learning_rate=LR, decay=DC)
    # es = callbacks.EarlyStopping(monitor=METRIC_VAR,verbose=1,mode=MODE,min_delta=0.001,patience=30);
    # mc = callbacks.ModelCheckpoint(MODEL_NAME, monitor=METRIC_VAR, mode=MODE, save_best_only=True,verbose=1);
    model = Sequential()
    model.add(Conv3D(32, (3, 3, x_train[0].shape[2]), padding='same',activation='relu', input_shape=SAMPLE_SHAPE))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Conv3D(32, (3, 3, x_train[0].shape[2]), padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(Conv3D(32, (3, 3, x_train[0].shape[2]), padding='same',activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',optimizer=OPT,metrics=['mae']);
    history = model.fit(x_train, y_train,epochs=EPOCHS,batch_size=SIZE,validation_data=(x_val, y_val),verbose=2)


#plot performance over time during training
def model_performance():
    global saved_model
    saved_model = tf.keras.models.load_model(MODEL_NAME)
    score = model.evaluate(x_test, y_test)
    #print(score)
    score1 = saved_model.evaluate(x_test, y_test)
    #print(score1)
    plt.subplot(211)  
    plt.plot(history.history['mae'])  
    plt.plot(history.history['val_mae'])  
    #plt.title('Model MAE')  
    plt.ylabel('MAE')  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='upper right')  
    plt.savefig(f'_{name}_EPOCHS_PERF.png')
    plt.show();
    #return saved_model

def results_df():
    global cnn_flower
    
    pred = saved_model.predict(x_test)
    cnn_flower = pd.DataFrame()   
    ##convert back response to oriignal scale
    cnn_flower['obs'] =  (y_test)#*1e49)**(1./20)
    cnn_flower['pred'] = (pred)#*1e49)**(1./20)
    cnn_flower['residuals'] = cnn_flower['pred']-cnn_flower['obs']
    cnn_flower.to_csv(f'{name}_.csv', sep='\t')
    dfs.append(cnn_flower)
    #return cnn_flower
    
def scatter_plot_results():
    
    plt.figure(figsize=(6,5))
    #first scatter plot
    x=np.linspace(0,120,121) 
    xy = np.vstack([cnn_flower['obs'],cnn_flower['pred']])
    z = gaussian_kde(xy)(xy)
    plt.scatter(cnn_flower['obs'], cnn_flower['pred'],marker='.')#, c = z,alpha = 0.7, marker = '.')
    #plt.title(f'{name}')
    plt.xlabel('Lodging score obs. (0-100%)')
    plt.ylabel('Lodging score pred. (0-100%)')
    plt.grid(color = '#D3D3D3', linestyle = 'solid')
    plt.plot(x,x,'k-',color="gray",linewidth=0.5) # identity line
    plt.ylim([0, 120])
    plt.xlim([0, 120])
    plt.savefig(f'{name}.png')
    
    
#subset target 5 dates around the lodging event

slices=[]
slices = (('2CNN_MULT1_0809_LODG',slice(30,35)),
        ('2CNN_MULT1_0815_LODG',slice(35,40)),
        ('2CNN_MULT1_0823_LODG',slice(40,45)),
        ('2CNN_MULT1_0828_LODG',slice(45,50)),
        ('2CNN_MULT1_0905_LODG',slice(50,55)))

list_data = []
labels = []              

for names, slicing in slices:
#for slicing in dates_bands_slices:\
    print(names)
    ##into list and rescaled
    #list_data.append(tuple[0])
    list_data.append(X[:,:,:,slicing])#/65536)
    labels.append(names);
    
for layers in list_data:
    print(layers.shape)
    
    
dfs = [] #outcome pred,obs, error

    
# Add argparse for choosing the model fit type
parser = argparse.ArgumentParser(description='Fit model for dataset.')
parser.add_argument('--fit-type', type=int, choices=[0, 1], default=0,
                    help='Select the type of model fitting (0 for 2D, 1 for 3D)')

args = parser.parse_args()

for name, dates in zip(labels, list_data):
    data_splitting(dates, y)
    # Run the selected fitting function based on the argument value
    if args.fit_type == 'time-point':
        fit_model()
    elif args.fit_type == 'temporal':
        fit_model1()
    else:
        print("Invalid fit type. Please choose 0 or 1.")
    
    model_performance()
    results_df()
    scatter_plot_results();