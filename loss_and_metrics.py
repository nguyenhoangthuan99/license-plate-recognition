# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:01:00 2019

@author: NguyenHoangThuan
"""

from keras import backend as K
import keras
import tensorflow as tf



def custom_loss(y_true, y_pred):

    s = K.shape(y_pred)
    y_true = K.reshape(y_true,(-1,s[-1]))
    y_pred = K.reshape(y_pred,(-1,s[-1]))
    loss = K.sum(keras.losses.categorical_crossentropy(y_true, y_pred))

    num = K.shape(y_true)[0]
    num=tf.cast(num,tf.float32)
    
    return K.mean(loss)/num

def char_acc(y_true, y_pred):

    s = K.shape(y_pred)	
    y_true_reshaped = K.reshape( y_true,  (-1,s[-1]) )
    y_pred_reshaped = K.reshape( y_pred,  (-1, s[-1]) )

	# correctly classified
    clf_pred =  K.argmax(y_pred_reshaped,axis = -1)
    y_true = K.argmax(y_true_reshaped,axis = -1)
    correct_pixels_per_class = K.cast( K.equal(clf_pred,y_true), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(s[:-1]), dtype='float32')
def image_acc(y_true, y_pred):

    s = K.shape(y_pred)
    clf_pred =  K.argmax(y_pred,axis = -1)
    y_true = K.argmax(y_true,axis = -1)
    correct_pixels_per_class = K.cast(K.all( K.equal(clf_pred,y_true),axis=-1), dtype='float32')

    return K.sum(correct_pixels_per_class) / K.cast(K.prod(s[0]), dtype='float32')