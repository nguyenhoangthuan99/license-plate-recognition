# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:04:30 2019

@author: NguyenHoangThuan
"""

import config as cf
from model import VGG
from keras.models import *
from prepare_data import *
import matplotlib.pyplot as plt
import random



train_x,train_y,val_x,val_y = create_data_test()

model = VGG(shape=(64, 256, 1))
model.summary()
model.load_weights(cf.CKP_PATH)

n = val_x.shape[0]

count = 0
pred = model.predict(val_x)
pred = np.argmax(pred,axis=-1)
true = np.argmax(val_y,axis = -1)
for i in range(n):
    if np.all(pred[i,:]==true[i]):
        count +=1
print("total acc: " +str(count/n*100)+"%")


k = random.randint(0,10000)  
imgplot = plt.imshow(val_x[k,:,:,:].reshape(64,256))  
arr = []
arr2 = []
for i in range(8):
    arr.append(dic[pred[k,i]])
    arr2.append(dic[true[k,i]])
print("predict: " +str("".join(arr))) 
print("ground truth label: "+str("".join(arr2)))
