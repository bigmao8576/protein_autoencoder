#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:36:45 2019

@author: bigmao
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import random

def file2dict(path,channel_num=2,norm=True):
    '''
    This function is used for the second dataset
    Convert the csv data to dictionary
    
    input:
        path: the path of the csv file
        channel_num: the channel number of each sample
        norm: If norm is Ture, each sample will be normalized accroding to it's own mean and std
    output:
        data_dict
        The length of all the samples
        
    
    '''
    data_frame=pd.read_csv(path, dtype={'Name':str, 'Value':float})
    data_list=data_frame.values.tolist()
    
    data_dict = {}
    for item in data_list:
        temp_data = np.array(item[1:])
        if norm:
            temp_data = (temp_data-np.mean(temp_data))/np.std(temp_data)
            
        length = len(temp_data)//channel_num
        
        data_dict[item[0]] = np.reshape(temp_data,[length,channel_num])
        
    return data_dict, length

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

def data2array(data):
    name_list = data.keys()
    name_list = list(name_list)
    
    train_data = np.array([data[item] for item in name_list])
    
    return np.array(train_data),name_list
        
def draw_pic_cost(plotdata,save_folder):

    plt.plot(plotdata["epoch"],np.log10(np.array(plotdata["train_cost"])),'b') 
    
    try:
        plt.plot(plotdata["epoch"],np.log10(np.array(plotdata["test_cost"])),'r')
        plt.legend(['Training _set','Testing_set'])
    except:
        pass
    
    plt.xlabel('Epoch')
    plt.ylabel('log10(cost)')
    file_name = os.path.join(save_folder,'cost.png')
    plt.savefig(file_name)
    plt.close()
    

def draw_pic_sample(x_train,train_result_output,save_folder):
    total_num = len(x_train)
            
    ind1 = random.randint(0,total_num-1)
    ind2 = random.randint(0,total_num-1)
    

    plt.figure(3)
    plt.plot(x_train[ind1,:,1]) 
    plt.plot(train_result_output[ind1,:,1]) 
    file_name = os.path.join(save_folder,'sampe1.png')
    plt.savefig(file_name)  
    plt.close()
    
    plt.figure(4)
    plt.plot(x_train[ind2,:,1]) 
    plt.plot(train_result_output[ind2,:,1]) 
    file_name = os.path.join(save_folder,'sampe2.png')
    plt.savefig(file_name)  
    plt.close()


def get_head(path):
    data_frame=pd.read_csv(path, dtype={'Name':str, 'Value':float})
    head_list = []
    for col in data_frame.columns: 
        head_list.append(col)
    
    return head_list[1:]