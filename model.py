# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 09:59:33 2019

@author: NguyenHoangThuan
"""

from keras.layers.core import *
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.regularizers import l2

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



