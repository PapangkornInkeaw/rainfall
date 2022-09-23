#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from calendar import monthrange
import datetime
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import math
import pickle
from os import path
import glob

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

tf.config.list_physical_devices('GPU')


# In[ ]:


def readTbb(xarrayData,lat,long,radius,step):
    latitude = np.concatenate([np.arange(lat-(step*radius),lat,step),np.arange(lat,lat+(step*radius)+(step/10),step)])
    longitude = np.concatenate([np.arange(long-(step*radius),long,step),np.arange(long,long+(step*radius)+(step/10),step)])
    #print(latitude)
    #print(longitude)
    return xarrayData.interp(latitude=latitude, longitude=longitude).get('tbb').values


# In[ ]:


def getDem(lat,lon):
    demPath = 'D:\Rain\ele\*.nc'
    list_of_files = glob.glob(demPath, recursive=True)
    for ncfile in list_of_files:
        x1 = xs.open_dataset(ncfile)
        latrange = x1.get('lat').values
        lonrange = x1.get('lon').values
        if lat >= min(latrange) and lat <= max(latrange) and lon >= min(lonrange) and lon <= max(lonrange):
            return x1.interp(lat=lat, lon=lon).get('ASTER_GDEM_DEM').values


# In[ ]:


def prepareData4Station(lat,long,radius,step,timestamp):
    data = np.empty((2*radius+1,2*radius+1,len(timestamp)+3))
    i=0
    for t in timestamp:
        year = t.year
        month = t.month
        day = t.day
        hour = t.hour
        minute = t.minute
        ncfiledir_h8 = r"Himawari-8, 9 User Data\2018\HS_H08_"+str(year)+str(month).zfill(2)+str(day).zfill(2)+"_"+str(hour).zfill(2)+str(minute).zfill(2)+"_B13_FLDK_R20.nc"
        ncfiledir_h9 = r"Himawari-8, 9 User Data\2018\HS_H09_"+str(year)+str(month).zfill(2)+str(day).zfill(2)+"_"+str(hour).zfill(2)+str(minute).zfill(2)+"_B13_FLDK_R20.nc"
        if path.exists(ncfiledir_h8):
            ncfiledir = ncfiledir_h8
        elif path.exists(ncfiledir_h9):
            ncfiledir = ncfiledir_h9
        else:
            ncfiledir = ''
            print('Warning: HS file is not available.')
        if path.exists(ncfiledir_h8) or path.exists(ncfiledir_h9):
            satData = xs.open_dataset(ncfiledir)
            latitude = np.concatenate([np.array([lat-x for x in step*np.arange(radius,0,-1)]),np.array([lat]),np.array([lat+x for x in step*np.arange(1,radius+1)])])
            longitude = np.concatenate([np.array([long-x for x in step*np.arange(radius,0,-1)]),np.array([long]),np.array([long+x for x in step*np.arange(1,radius+1)])])
            data[:,:,i] = satData.interp(latitude=latitude, longitude=longitude).get('tbb').values
        else:
            data[:,:,i] = np.zeros((2*radius+1,2*radius+1))
        i = i+1

    data[:,:,i] = lat*np.ones((2*radius+1,2*radius+1))
    i = i+1
    data[:,:,i] = long*np.ones((2*radius+1,2*radius+1))
    i = i+1
    dem = getDem(lat,long)
    data[:,:,i] = dem*np.ones((2*radius+1,2*radius+1))
    return data


# In[ ]:


rainfallData = pd.read_csv('dailyRainfallData2018.csv')


# In[ ]:


def runExpriment(year, month, day):
    radius = 2
    step = 0.010000
    hour = 7
    minute = 0
    stratTime = datetime.datetime(year, month, day, hour,minute)
    timestamp = [stratTime+datetime.timedelta(minutes=x) for x in range(0,1441,30)]
    date = str(day)+'-'+str(month)+'-'+str(year)
    selectIdx = rainfallData.index[rainfallData[date] >= 0.0].tolist()
    dsx = np.empty((len(selectIdx),2*radius+1,2*radius+1,len(timestamp)+3))
    dsy = np.empty(len(selectIdx))
    i = 0
    print("There are ",len(selectIdx)," stations.")
    for idx in selectIdx:
        print("Preparing data for station ",idx)
        lat = rainfallData.iloc[idx].lat
        long = rainfallData.iloc[idx].long
        dsx[i,:,:,:] = prepareData4Station(round(lat,6),round(long,6),radius,step,timestamp)
        dsy[i] = rainfallData.iloc[idx][date]    
        i = i+1

        x_train, x_test, y_train, y_test = train_test_split(dsx, dsy, test_size=0.3)
    
    # Run experiment
    epochs=5000
    number_of_trial = 30
    ypredict_result = np.empty((number_of_trial,y_test.shape[0]))
    ytest_result = np.empty((number_of_trial,y_test.shape[0]))
    xtest_result = np.empty((number_of_trial,x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    xtrain_result = np.empty((number_of_trial,x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))

    for trial in range(0,number_of_trial):
        x_train, x_test, y_train, y_test = train_test_split(dsx, dsy, test_size=0.3)
        trainingDataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        trainingDataset = trainingDataset.shuffle(100).batch(32)
        testDataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        testDataset = testDataset.batch(32)
        cnnModel = Sequential([
            layers.Conv2D(32, 3, padding='same', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])),
            layers.Activation('relu'),
            layers.Conv2D(64, 3, padding='valid'),
            layers.Activation('relu'),
            layers.Conv2D(128, 3, padding='valid'),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dense(256, activation='sigmoid'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='relu'),
        ])
        sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.7, nesterov=True)
        cnnModel.compile(optimizer=sgd,
                      loss='mse',
                      metrics=['mse'])
        #with tf.device('/gpu:1'):    
        history = cnnModel.fit(
            trainingDataset,
            validation_data=(testDataset),
            batch_size=32,
            epochs=epochs,
            verbose=0
        )
        y_pred = cnnModel.predict(x_test)
        ypredict_result[trial,:] = y_pred.reshape(y_pred.shape[0])
        ytest_result[trial,:] = y_test
        xtest_result[trial,:] = x_test
        xtrain_result[trial,:] = x_train
        print("Trial",trial," MSE = ",mean_squared_error(y_test, y_pred))
    
    # Save results
    MSE = np.empty(number_of_trial)
    MAE = np.empty(number_of_trial)
    RMSE = np.empty(number_of_trial)
    r_2 = np.empty(number_of_trial)
    for trial in range(0,number_of_trial):
        MSE[trial] = mean_squared_error(ytest_result[trial,:], ypredict_result[trial,:])
        MAE[trial] = mean_absolute_error(ytest_result[trial,:], ypredict_result[trial,:])
        RMSE[trial] = mean_squared_error(ytest_result[trial,:], ypredict_result[trial,:], squared=False)
        r_2[trial] = r2_score(ytest_result[trial,:], ypredict_result[trial,:])
    print("MSE = ",np.mean(MSE))
    print("MAE = ",np.mean(MAE))
    print("RMSE = ",np.mean(RMSE))


# In[ ]:


experiment_date = ['23/04/2018','07/05/2018','21/05/2018','04/06/2018','18/06/2018','02/07/2018','16/07/2018','30/07/2018','13/08/2018','27/08/2018','10/09/2018','24/09/2018','08/10/2018','22/10/2018','05/11/2018','19/11/2018','03/12/2018','17/12/2018','30/12/2018']
for ex in experiment_date:
    date_time_obj = datetime.datetime.strptime(ex, '%d/%m/%Y')
    year = date_time_obj.year
    month = date_time_obj.month
    day = date_time_obj.day
    print('Running experiment for',day,'-',month,'-',year)
    
    runExpriment(year, month, day)

