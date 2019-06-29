# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 09:48:16 2019

@author: NguyenHoangThuan
"""

import pandas as pd
import numpy as np
import cv2
import config as cf
from keras.utils.np_utils import to_categorical   

letters = " ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"
dic = {}
for i in range(len(letters)):
    dic[i] = letters[i]
invert_dic = {}
for i in range(len(letters)):
    invert_dic[letters[i]] = i

db = pd.read_csv(cf.PATH_TO_DATASET_CSV_FILE)
base_dir = cf.PATH_TO_DATASET_FOLDER
n = len(db)
def create_data_train(db =db, base_dir = base_dir, num_sample = 4*n//5,invert_dic = invert_dic):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for i in range(num_sample):
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
    return train_x,train_y,val_x,val_y

def create_data_test(db =db, base_dir = base_dir, num_sample = n,invert_dic = invert_dic):
 
    val_x = []
    val_y = []
    for i in range(num_sample):
        temp_y= np.zeros((8))          
        if db["train"][i]==0:
            temp_x = cv2.imread(base_dir+db["image_path"][i], cv2.IMREAD_GRAYSCALE)
            temp_x = cv2.resize(temp_x,(256,64))
               
            val_x.append(temp_x)
            for j,k in enumerate(db["lp"][i]):
                temp_y[j] = invert_dic[k] 

            val_y.append(temp_y) 
 
    val_x = np.array(val_x).reshape(-1,64,256,1)
    val_y = np.array(val_y)
    val_y = to_categorical(val_y,36)
    
    val_x = val_x/255
    return val_x,val_y