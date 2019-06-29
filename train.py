# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:06:03 2019

@author: NguyenHoangThuan
"""

from keras.preprocessing.image import *
from prepare_data import *
from loss_and_metrics import *
from model import VGG
from keras.models import *
from keras.callbacks import *
import config as cf
train_x,train_y,val_x,val_y = create_data()

model = VGG(shape=(64, 256, 1))
model.summary()

model.compile(loss = custom_loss,
                  optimizer='adam',
                  metrics=[char_acc,image_acc])

datagen = ImageDataGenerator(width_shift_range=0.14,
                                 height_shift_range=0.08,
                                 fill_mode='constant',
                                 zoom_range = 0.1,
                                 rotation_range = 10,
                                 #rescale  =1./255
                                 )
mcp_save = ModelCheckpoint(cf.CKP_PATH, save_best_only=True, monitor='val_loss', mode='min',verbose=1)

def scheduler(epoch):
    if epoch <4 :
        return 0.001
    elif epoch < 10:
        return 0.001/5
    elif epoch < 15:
        return  0.0001
    elif epoch <30:
        return  0.0001/2
n = train_x.shape[0] 
#model.load_weights("plate.h5")   
lr_reduce = LearningRateScheduler(scheduler,verbose = 1)
model.fit_generator(datagen.flow(train_x, train_y,batch_size=cf.BATCH_SIZE),
                         epochs = 10,
                         steps_per_epoch=n//cf.BATCH_SIZE,
                         callbacks=[lr_reduce,mcp_save],
                         validation_data=(val_x, val_y))   
