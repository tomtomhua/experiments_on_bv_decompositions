#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np

from pmlb import fetch_data, classification_dataset_names
from scipy import stats

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import random as rd
import math

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from pmlb import fetch_data, classification_dataset_names
adult_data = fetch_data('adult')
print(len(classification_dataset_names))
print(adult_data.describe())


# In[3]:


X,y = fetch_data('adult',return_X_y = True) 
train_X, test_X, train_y, test_y = train_test_split(X, y)


# In[8]:


##get the best classfier
def ETC_best(X,y):
    depth =  X.shape[1]
    param ={'n_estimators':range(10,71,10),'max_features':[0.05,0.1,0.15,0.2,0.3],'min_samples_split':[2,0.01,0.02,0.05,0.1],'min_samples_leaf':[1,0.005,0.01,0.02,0.05],'max_depth':[depth,int(math.sqrt(depth)),int(math.log2(depth))]}
    gsearch= GridSearchCV(estimator =ExtraTreesClassifier(random_state=10), 
                       param_grid =param,scoring='accuracy',cv=5, n_jobs = 5)
    gsearch.fit(X,y)
    best = gsearch.best_params_
    ETC_best = ExtraTreesClassifier(random_state=10,n_estimators = best["n_estimators"],max_features = best["max_features"],min_samples_split = best["min_samples_split"],
                                    min_samples_leaf = best["min_samples_leaf"])
    return(ETC_best)



# In[12]:


def Extra_err_bias_var_clc_def(X,y,n = 100,train_size_p = 0.6,pool_size_p = 0.15,test_size_p = 0.15):
    pool_X, test_X, pool_y, test_y = train_test_split(X, y,train_size = train_size_p ,test_size = test_size_p)
    clc = ExtraTreesClassifier(n_jobs = 6,random_state = 10)
    clc_best =  ETC_best(pool_X,pool_y)
    clc_best.n_jobs = 6
    clc_best.random_state = 10
    train_size = math.ceil(len(y)*pool_size_p)
    test_size = len(test_y)
    num_generate = n
    all_pred = np.zeros((num_generate,test_size),dtype = np.int)
    train_pred = np.zeros((num_generate,train_size),dtype = np.int)
    train_err = np.zeros(num_generate,dtype = np.float)
    test_err = np.zeros(num_generate,dtype = np.float)
    
    
    all_pred_best = np.zeros((num_generate,test_size),dtype = np.int)
    train_pred_best = np.zeros((num_generate,train_size),dtype = np.int)
    train_err_best = np.zeros(num_generate,dtype = np.float)
    test_err_best = np.zeros(num_generate,dtype = np.float)
    
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
    
        fit_model_best = clc_best.fit(learn_X,learn_y)
        train_pred_best[i] = fit_model_best.predict(learn_X)
        train_err_best[i] = (train_pred_best[i] != learn_y).mean()
        pred_best = fit_model_best.predict(test_X)
        all_pred_best[i] = pred_best
        test_err_best[i] = (pred_best!= test_y).mean()
        
    avr_test_err = np.mean(test_err)
    std_test_err = np.std(test_err)
    avr_train_err = np.mean(train_err)
    std_train_err = np.std(train_err)
    
    avr_test_best_err = np.mean(test_err_best)
    std_test_best_err = np.std(test_err_best)
    avr_train_best_err = np.mean(train_err_best)
    std_train_best_err = np.std(train_err_best)
    
    main_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred)
    bias = sum(test_y != main_pred)/len(test_y)
    Var=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var[i]=sum(all_pred[i]!=main_pred)/len(main_pred)
    var=Var.mean()
    stat_out = [bias,var,avr_train_err,std_train_err,avr_test_err,std_test_err]
    
    main_pred_best = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis = 0, arr = all_pred_best)
    bias_best = sum(test_y != main_pred_best)/len(test_y)
    Var_best=np.zeros(num_generate, dtype=np.float)
    for i in range(num_generate):
        Var_best[i]=sum(all_pred_best[i]!=main_pred_best)/len(main_pred_best)
    var_best=Var_best.mean()
    error_best = bias_best + var_best
    stat_out_best = [bias_best,var_best,avr_train_best_err,std_train_best_err,avr_test_best_err,std_test_best_err]
    return [stat_out,stat_out_best]


# In[10]:


def bvr_cal(train,pool,test):
    bias_variance_err = []
    for classification_dataset in classification_dataset_names:
        X, y = fetch_data(classification_dataset, return_X_y=True)
        y = y - min(y) # exists y = -1 etc
        stat = Extra_err_bias_var_clc_def(X,y,100,train,pool,test)
        bias_variance_err.append(stat)
    return bias_variance_err


# In[9]:


###test1
##bias_variance_err = []
#X,y = fetch_data('adult',return_X_y = True) 
#y = y - min(y) # exists y = -1 etc
#stat = Extra_err_bias_var_clc_def(X,y,100)
#bias_variance_err.append(stat)
#bias_variance_err


# In[ ]:


stat_d_b = bvr_cal(0.6,0.15,0.15)


# In[ ]:


np.save("outbd.npy",stat_d_b)

