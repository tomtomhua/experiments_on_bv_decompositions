#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pmlb import fetch_data, classification_dataset_names
from scipy import stats

import random as rd
import math

import warnings
warnings.filterwarnings('ignore')


# In[15]:


def robust(y,rate = 0.05):
   index = rd.sample(range(0,len(y)),k = int(len(y)*rate))
   maxy = max(y)
   miny = min(y)
   for i in index:
       if(y[i] == maxy):
           y[i] = miny
       else:
           y[i] = maxy
   return(y)


# In[16]:


def delete(X,y,rate = 0.05):
    index = rd.sample(range(0,len(y)),k = int(len(y)*rate))
    y0 = np.delete(y,np.array(index))
    X0 = np.delete(X,np.array(index),0)
    return([X0,y0])


# In[26]:


def Extra_err_bias_var_robust(X,y,n = 100,train_size_p = 0.6,pool_size_p = 0.15,test_size_p = 0.15) :
    pool_X, test_X, pool_y, test_y = train_test_split(X, y,train_size = train_size_p ,test_size = test_size_p)
    train_size = math.ceil(len(y)*pool_size_p)
    test_size = len(test_y)
    num_generate = n
    all_pred = np.zeros((num_generate,test_size),dtype = np.int)
    all_pred_r = np.zeros((num_generate,test_size),dtype = np.int)
    all_pred_d = np.zeros((num_generate,test_size),dtype = np.int)
    
    train_err = np.zeros(num_generate,dtype = np.float)
    test_err = np.zeros(num_generate,dtype = np.float)
    
    train_err_r = np.zeros(num_generate,dtype = np.float)
    test_err_r = np.zeros(num_generate,dtype = np.float)
    
    train_err_d = np.zeros(num_generate,dtype = np.float)
    test_err_d = np.zeros(num_generate,dtype = np.float)
    
    for i in range(0,num_generate):
        index = rd.sample(range(0,len(pool_y)) ,k = train_size)
        learn_X = pool_X[index]
        learn_y = pool_y[index]
        learn_yr = robust(learn_y)
        learn_Xd,learn_yd = delete(learn_X,learn_y)
        
        ETC = ExtraTreesClassifier(n_jobs = 6)
        ETC_r = ExtraTreesClassifier(n_jobs = 6)
        ETC_d = ExtraTreesClassifier(n_jobs = 6)
        
        fit_ETC = ETC.fit(learn_X,learn_y)
        fit_ETC_r = ETC.fit(learn_X,learn_yr)
        fit_ETC_d = ETC.fit(learn_Xd,learn_yd)
        
        train_err[i] = (fit_ETC.predict(learn_X) != learn_y).mean()
        train_err_r[i] = (fit_ETC_r.predict(learn_X) != learn_yr).mean()
        train_err_d[i] = (fit_ETC_d.predict(learn_Xd) != learn_yd).mean()
        
        pred = fit_ETC.predict(test_X)
        pred_r = fit_ETC_r.predict(test_X)
        pred_d = fit_ETC_d.predict(test_X)
        
        test_err[i] = (pred!= test_y).mean()
        test_err_r[i] = (pred_r!= test_y).mean()
        test_err_d[i] = (pred_d!= test_y).mean()
        
        all_pred[i] = pred
        all_pred_r[i] = pred_r
        all_pred_d[i] = pred_d

             
    avr_test_err = np.mean(test_err)
    std_test_err = np.std(test_err)
    avr_train_err = np.mean(train_err)
    std_train_err = np.std(train_err)
    
    avr_test_r_err = np.mean(test_err_r)
    std_test_r_err = np.std(test_err_r)
    avr_train_r_err = np.mean(train_err_r)
    std_train_r_err = np.std(train_err_r)
        
    avr_test_d_err = np.mean(test_err_d)
    std_test_d_err = np.std(test_err_d)
    avr_train_d_err = np.mean(train_err_d)
    std_train_d_err = np.std(train_err_d)
    
    main_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred)
    main_pred_r = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred_r)
    main_pred_d = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred_d)
    
    bias = sum(test_y != main_pred)/len(test_y)
    Var=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var[i]=sum(all_pred[i]!=main_pred)/len(main_pred)
    var=Var.mean()
    stat_out = [bias,var,avr_train_err,std_train_err,avr_test_err,std_test_err]
    
    bias_r = sum(test_y != main_pred_r)/len(test_y)
    Var_r=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var_r[i]=sum(all_pred_r[i]!=main_pred_r)/len(main_pred_r)
    var_r=Var_r.mean()
    stat_out_r = [bias_r,var_r,avr_train_r_err,std_train_r_err,avr_test_r_err,std_test_r_err]
    
    
    bias_d = sum(test_y != main_pred_d)/len(test_y)
    Var_d=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var_d[i]=sum(all_pred_d[i]!=main_pred_d)/len(main_pred_d)
    var_d=Var_d.mean()
    stat_out_d = [bias_d,var_d,avr_train_d_err,std_train_d_err,avr_test_d_err,std_test_d_err]
    return [stat_out,stat_out_r,stat_out_d]


# In[5]:


def bvr_cal(train,pool,test):
    bias_variance_err = []
    for classification_dataset in classification_dataset_names:
        X, y = fetch_data(classification_dataset, return_X_y=True)
        y = y - min(y) # exists y = -1 etc
        stat = Extra_err_bias_var_robust(X,y,100,train,pool,test)
        bias_variance_err.append(stat)
    return bias_variance_err


# In[30]:


c1 = np.repeat(0.3,9)
c2 = np.repeat([0.1,0.15,0.2],3)
c3 = np.tile([0.1,0.15,0.2],3)
size_set = np.column_stack((c1,c2,c3))
numsize = size_set.shape[0]


# In[ ]:


size_bvr = np.zeros((numsize,166,3,6))
for i in range(0,numsize):
    train,pool,test = size_set[i]
    size_bvr[i] = bvr_cal(train,pool,test)
###to run later 


# In[ ]:


np.save("outbv3.npy",size_bvr[range(0,2)])


# In[ ]:




