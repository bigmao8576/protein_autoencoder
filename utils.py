#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:36:45 2019

@author: bigmao
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def file2dict(path,channel_num=2):
    '''
    This function is used for the second dataset
    Convert the csv data to dictionary
    
    input:
        path: the path of the csv file
        channel_num: the channel number of each sample
    output:
        data_dict
        The length of all the samples
    '''
    data_frame=pd.read_csv(path, dtype={'Name':str, 'Value':float})
    data_list=data_frame.values.tolist()
    
    data_dict = {}
    for item in data_list:
        temp_data = np.array(item[1:])
        data_dict[item[0]] = np.reshape(temp_data,[len(temp_data)//channel_num,channel_num])
        
    return data_dict, len(temp_data)//2

def data2fold(data,fold = 5):
    name_list = data.keys()
    name_list = list(name_list)
    
    kf = KFold(n_splits=fold,shuffle=True)
    kf.get_n_splits(name_list)        
    
    fold_num = 1
    fold_dict = {}
    tot_fold_name = []
    
    for train_ind, test_ind in kf.split(name_list):
    
        train_name = [name_list[i] for i in train_ind]
        test_name = [name_list[i] for i in test_ind]
        
        
        train_data = np.array([data[item] for item in train_name])
        test_data = np.array([data[item] for item in test_name])
        
        fold_name = 'fold_%d'%fold_num
        
        fold_num +=1
        
        fold_dict[fold_name]={'train_name':train_name,
                              'test_name':test_name,
                              'train_data':train_data,
                              'test_data':test_data}
        tot_fold_name.append(fold_name)
    return fold_dict,tot_fold_name
        