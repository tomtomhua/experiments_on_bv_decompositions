#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

from pmlb import fetch_data, classification_dataset_names
from scipy import stats

import random as rd
import math

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pmlb import fetch_data, classification_dataset_names
adult_data = fetch_data('adult')
print(len(classification_dataset_names))
print(adult_data.describe())


# In[20]:


X,y = fetch_data('adult',return_X_y = True) 
train_X, test_X, train_y, test_y = train_test_split(X, y)


# In[21]:


X.shape


# In[12]:



# param ={'n_estimators':range(10,71,10),
#         'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
#         'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
#         'min_data_in_leaf':[1,0.005,0.01,0.02,0.05],
#         'max_depth':[depth,int(math.sqrt(depth)),int(math.log2(depth))]}


# In[15]:


#lgb_clc = lgb_best(X,y)
# lgb_clc.score(X,y)


# In[ ]:





# In[6]:


#max_depth
#num_leaves
#min_data_in_leaf
#feature_fraction
#bagging_fraction


# In[64]:


def LGB_err_bias_var_clc(X,y,clc,n = 100,train_size_p = 0.6,pool_size_p = 0.15,test_size_p = 0.15):
    pool_X, test_X, pool_y, test_y = train_test_split(X, y,train_size = train_size_p ,test_size = test_size_p)
    ##Suppose i have a par dict
    train_size = math.ceil(len(y)*pool_size_p)
    test_size = len(test_y)
    num_generate = n
    all_pred = np.zeros((num_generate,test_size),dtype = np.int)
    train_pred = np.zeros((num_generate,train_size),dtype = np.int)
    train_err = np.zeros(num_generate,dtype = np.float)
    test_err = np.zeros(num_generate,dtype = np.float)
    for i in range(0,num_generate):
        index = rd.sample(range(0,len(pool_y)) ,k = train_size)
        learn_X = pool_X[index]
        learn_y = pool_y[index]
        fit_model = clc.fit(learn_X,learn_y)
        train_pred[i] = fit_model.predict(learn_X)
        train_err[i] = (train_pred[i] != learn_y).mean()
        
        pred = fit_model.predict(test_X)
        all_pred[i] = pred
        test_err[i] = (pred!= test_y).mean()
        
        
    avr_test_err = np.mean(test_err)
    std_test_err = np.std(test_err)
    avr_train_err = np.mean(train_err)
    std_train_err = np.std(train_err)

    main_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred)
    bias = sum(test_y != main_pred)/len(test_y)
    Var=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var[i]=sum(all_pred[i]!=main_pred)/len(main_pred)
    var=Var.mean()
    stat_out = [bias,var,avr_train_err,std_train_err,avr_test_err,std_test_err]
    return stat_out


# In[27]:


# param ={'n_estimators':range(10,71,10),
#         'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
#         'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
#         'min_data_in_leaf':[1,0.005,0.01,0.02,0.05],
#         'max_depth':[depth,int(math.sqrt(depth)),int(math.log2(depth))]}

def LGB_mul_par_com_clc(X,y,n = 100,train_size_p = 0.6,pool_size_p = 0.15,test_size_p = 0.15):
    stat_n_estimators = []
    num_sample = X.shape[0]
    for i in range(10,70,10):
        clc = lgb.LGBMClassifier(random_state = 10,n_jobs = 6,n_estimators = i)
        stat_n_estimators_i = LGB_err_bias_var_clc(X,y,clc)
        stat_n_estimators.append(stat_n_estimators_i)
    stat_feature_fraction = []
    for i in [0.6,0.7,0.8,0.9,1.0]:
        clc = lgb.LGBMClassifier(random_state = 10,n_jobs = 6,feature_fraction = i)
        stat_feature_fraction_i = LGB_err_bias_var_clc(X,y,clc)
        stat_feature_fraction.append(stat_feature_fraction_i)
    stat_bagging_fraction = []
    for i in [0.6,0.7,0.8,0.9,1.0]:
        clc = lgb.LGBMClassifier(random_state = 10,n_jobs = 6,bagging_fraction = i)
        stat_bagging_fraction_i = LGB_err_bias_var_clc(X,y,clc)
        stat_bagging_fraction.append(stat_bagging_fraction_i)
    stat_min_data_in_leaf = []
    for i in [int(0.005*num_sample)+1,int(0.01*num_sample)+1,int(0.02*num_sample)+1
              ,int(0.05*num_sample)+1,1]:
        clc = lgb.LGBMClassifier(random_state = 10,n_jobs = 6,min_data_in_leaf = i)
        stat_min_data_in_leaf_i = LGB_err_bias_var_clc(X,y,clc)
        stat_min_data_in_leaf.append(stat_min_data_in_leaf_i)
    stat_max_depth = []
    depth =  X.shape[1]
    for i in [depth,int(math.sqrt(depth)),int(math.log2(depth))]:
        clc = lgb.LGBMClassifier(random_state = 10,n_jobs = 6,max_depth = i)
        stat_max_depth_i = LGB_err_bias_var_clc(X,y,clc)
        stat_max_depth.append(stat_max_depth_i)
    stat_out = [stat_n_estimators,stat_feature_fraction,stat_bagging_fraction,stat_min_data_in_leaf,stat_max_depth]
    return(stat_out)


# In[ ]:


def bvr_cal(train,pool,test):
    bias_variance_err = []
    for classification_dataset in classification_dataset_names:
        X, y = fetch_data(classification_dataset, return_X_y=True)
        y = y - min(y) # exists y = -1 etc
        if max(y)==1 :
            stat =LGB_mul_par_com_clc(X,y,train,pool,test)
            bias_variance_err.append(stat)
    return bias_variance_err


# In[ ]:


stat_p_c = bvr_cal(0.6,0.15,0.15)


# In[ ]:


np.save("lgbpc.npy",stat_p_c)

