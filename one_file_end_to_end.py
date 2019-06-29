#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:00:42 2019

@author: tuanna118
"""


from keras.preprocessing.image import *
from keras.layers.core import *
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras
import numpy as np
from keras import backend as K
import pandas as pd
from keras.callbacks import *
np.random.seed(0)
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical   
from keras.regularizers import l2
import cv2

letters = " ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
dic = {}
for i in range(len(letters)):
    dic[i] = letters[i]
invert_dic = {}
for i in range(len(letters)):
    invert_dic[letters[i]] = i

db = pd.read_csv("/media/tuanna118/DATA/trainVal.csv")
base_dir = "/media/tuanna118/DATA/data/"
n = len(db)
train_x = []
train_y = []
val_x = []
val_y = []
for i in range(4*n//5):
    temp_y= np.zeros((8))
    if db["train"][i]==1:
        path = base_dir + db["image_path"][i]
        temp_x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        temp_x = cv2.resize(temp_x,(256,64))
        #temp_x = temp_x/255
        train_x.append(temp_x)
        for j,k in enumerate(db["lp"][i]):
            temp_y[j] = invert_dic[k] 
        
#        temp_y = to_categorical(temp_y,36) 
#        db["y_col"][i] = temp_y  
        train_y.append(temp_y)    
    if db["train"][i]==0:
        temp_x = cv2.imread(base_dir+db["image_path"][i], cv2.IMREAD_GRAYSCALE)
        temp_x = cv2.resize(temp_x,(256,64))
        #temp_x = temp_x/255
        val_x.append(temp_x)
        for j,k in enumerate(db["lp"][i]):
            temp_y[j] = invert_dic[k] 
#        temp_y = to_categorical(temp_y,36) 
#        db["y_col"][i] = temp_y   
        val_y.append(temp_y) 
train_x = np.array(train_x).reshape(-1,64,256,1)
train_y = np.array(train_y)
val_x = np.array(val_x).reshape(-1,64,256,1)
val_y = np.array(val_y)
val_y = to_categorical(val_y,36)
train_y = to_categorical(train_y,36)
train_x = train_x /255
val_x = val_x/255
def VGG(shape=(64, 256, 1),n_channels=64,weight_decay=0,batch_momentum=0.99):
    bn_axis = 3
    input_ = Input(shape=shape)
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input_)
    x = BatchNormalization(axis=bn_axis, name='bn00_x1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn01_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn11_x1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn12_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn21_x1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn22_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
#    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3', kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(x)
#    x = BatchNormalization(axis=bn_axis, name='bn23_x3', momentum=batch_momentum)(x)
#    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn31_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn32_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
#    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3', kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(x)
#    x = BatchNormalization(axis=bn_axis, name='bn33_x2', momentum=batch_momentum)(x)
#    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn41_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
#    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(x)
#    x = BatchNormalization(axis=bn_axis, name='bn42_x2', momentum=batch_momentum)(x)
#    x = Activation('relu')(x)
    
    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv3', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn43_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(1024, (3, 3), padding='same', name='block6_conv1', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn51_x2', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1024*2, (3, 3), padding='same', name='block6_conv12', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis, name='bn51_x22', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x= Dropout(0.3, noise_shape=None, seed=None)(x)
   
    
    #block5
   
    
    X = AveragePooling2D((2, 2), strides = (2, 1), name='avg_pool1',padding ='same')(x)
    X = Reshape((8,1024*2))(X)
    
    
    X = Conv1D(512, 3, strides=1, padding='same',name = 'conv1y'  ,activation=None,  dilation_rate=1,  use_bias=True, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.000))(X)
    X = BatchNormalization(axis = 2, name = 'bn01y')(X)
    X = Activation('relu')(X)
    X= Dropout(0.3, noise_shape=None, seed=None)(X)
    X = Conv1D(36, 1 , strides=1, padding='same',name = 'conv1x'  ,activation=None,  dilation_rate=1,  use_bias=True, kernel_initializer="he_normal")(X)
    X = BatchNormalization(axis = 2, name = 'bnhe')(X)
    X = Activation('softmax')(X)
    model = Model(inputs = [input_], outputs = [X])
    return model

model = VGG(shape=(64, 256, 1))

def custom_loss(y_true, y_pred):
#    softmax = K.softmax(y_pred)
    s = K.shape(y_pred)
    y_true = K.reshape(y_true,(-1,s[-1]))
    y_pred = K.reshape(y_pred,(-1,s[-1]))
    loss = K.sum(keras.losses.categorical_crossentropy(y_true, y_pred))
    #K.mean(keras.losses.categorical_crossentropy(y_true, y_pred))
    num = K.shape(y_true)[0]
    num=tf.cast(num,tf.float32)
    
    return K.mean(loss)/num

def pixel_acc(y_true, y_pred):
#    y_true = y_true[:,:,:,0]
#    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    s = K.shape(y_pred)
    
	# reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape( y_true,  (-1,s[-1]) )
    y_pred_reshaped = K.reshape( y_pred,  (-1, s[-1]) )

	# correctly classified
    clf_pred =  K.argmax(y_pred_reshaped,axis = -1)
    y_true = K.argmax(y_true_reshaped,axis = -1)
    correct_pixels_per_class = K.cast( K.equal(clf_pred,y_true), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(s[:-1]), dtype='float32')
def acc1(y_true, y_pred):
#    y_true = y_true[:,:,:,0]
#    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    s = K.shape(y_pred)
    
	# reshape such that w and h dim are multiplied together
    #y_true_reshaped = K.reshape( y_true,  (-1,s[-1]) )
    #y_pred_reshaped = K.reshape( y_pred,  (-1, s[-1]) )

	# correctly classified
    clf_pred =  K.argmax(y_pred,axis = -1)
    y_true = K.argmax(y_true,axis = -1)
    correct_pixels_per_class = K.cast(K.all( K.equal(clf_pred,y_true),axis=-1), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(s[0]), dtype='float32')
model.compile(loss = custom_loss,
                  optimizer='adam',
                  metrics=[pixel_acc,acc1])
model.summary()


datagen = ImageDataGenerator(width_shift_range=0.14,
                                 height_shift_range=0.08,
                                 fill_mode='constant',
                                 zoom_range = 0.1,
                                 rotation_range = 10,
                                 #rescale  =1./255
                                 )
mcp_save = ModelCheckpoint('plate2.h5', save_best_only=True, monitor='val_loss', mode='min',verbose=1)

def scheduler(epoch):
    if epoch <3 :
        return 0.001/5
    elif epoch < 10:
        return 0.001/10
    elif epoch < 15:
        return  0.00001
    elif epoch <30:
        return  0.00001/2
n = train_x.shape[0] 
model.load_weights("plate.h5")   
lr_reduce = LearningRateScheduler(scheduler,verbose = 1)
model.fit_generator(datagen.flow(train_x, train_y,batch_size=64),
                         epochs = 10,
                         steps_per_epoch=n//64,
                         callbacks=[lr_reduce,mcp_save],
                         validation_data=(val_x, val_y))   

#model.load_weights("plate.h5")
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import random
#n = val_x.shape[0]
#count = 0
#pred = model.predict(val_x)
#pred = np.argmax(pred,axis=-1)
#true = np.argmax(val_y,axis = -1)
#for i in range(n):
#    if np.all(pred[i,:]==true[i]):
#        count +=1
#print(count/n)
#k = random.randint(0,10000)  
#imgplot = plt.imshow(val_x[k,:,:,:].reshape(64,256))  
#arr = []
#for i in range(8):
#    arr.append(dic[pred[k,i]])
#print(arr)    
#0.9757237083180652 plate.h5
#0.9758153169659216