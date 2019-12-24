
"""
This script is used for summerizing the results after the 5-fold validation is trained.

Output files:
    recovered sequences (output of decoder) from the training samples
    recovered sequences (output of encoder) from the testing samples
    (note that the input of encoder was normalized based on individual instance for data set 2)
    
    vector representation of the training samples
    vector representation of the testing samples.
    
"""

import numpy as np
import pickle
import os
from copy import deepcopy
import utils
import ops
import argparse
import sys
import random
import pandas as pd


# first, load the original data
path = 'data_rhythm.csv'
head_list = utils.get_head(path)
#import the original data
data_dict,_ = utils.file2dict(path,norm=False)

# load the original fold data
fold_dict = pickle.load(open('fold_dict.pkl','rb'), encoding='latin1')

# locate the result folder
result_folder = 'fold_results_h40_bs100_lr0.00100'


for fold_name in range(1,6):
    
    # load the training results in this fold
    fold_data = fold_dict['fold_%d'%fold_name]
    
    result_folder_fold_path = os.path.join(result_folder,'fold_%d'%fold_name)
    
    if not os.path.exists(result_folder_fold_path):
        raise FileNotFoundError(result_folder_fold_path + 'does not exists')
    
    fold_result = pickle.load(open(os.path.join(result_folder_fold_path,'plotdata.pkl'),'rb'), encoding='latin1')
    
    
    x_train=fold_data['train_data']
    x_test=fold_data['test_data']
    
    x_train = np.reshape(x_train,[x_train.shape[0],x_train.shape[1]*x_train.shape[2]])
    x_test = np.reshape(x_test,[x_test.shape[0],x_test.shape[1]*x_test.shape[2]])
    
    x_train_rec = fold_result['train_recover_data']
    x_test_rec = fold_result['test_recover_data']
    
    
    x_train_rec = np.reshape(x_train_rec,[x_train_rec.shape[0],x_train_rec.shape[1]*x_train_rec.shape[2]])
    x_test_rec = np.reshape(x_test_rec,[x_test_rec.shape[0],x_test_rec.shape[1]*x_test_rec.shape[2]])
    
    train_name = fold_data['train_name']
    test_name = fold_data['test_name']
    
    train_rep = fold_result['train_representation']
    test_rep = fold_result['test_representation']
    
    # now let's begin with the training data
    x_train_rec_reverse = []
    x_train_original = []
    
    
    for name, temp_rec in zip(train_name,x_train_rec):
        temp_true = data_dict[name]
        temp_true = np.reshape(temp_true,[1,temp_true.shape[0]*temp_true.shape[1]])  
        
        temp_mean = np.mean(temp_true)
        temp_std = np.std(temp_true)
        
        temp_true_norm = (temp_true-temp_mean)/temp_std
        
        x_train_rec_reverse.append(temp_rec*temp_std+temp_mean)    
        x_train_original.append(temp_true)
    
    
    # now writing the recovered data, which have the same scale of the original data
    df_re_data_reverse=pd.DataFrame(x_train_rec_reverse,index=train_name)
    file_name = os.path.join(result_folder_fold_path,'train_recovered_data_reverse.csv')
    df_re_data_reverse.to_csv(file_name,header=head_list)
    
    
    # now writing the recovered data, which is equivalent to the normalized sample 
    df_re_data=pd.DataFrame(x_train_rec,index=train_name)
    file_name = os.path.join(result_folder_fold_path,'train_recovered_data.csv')
    df_re_data.to_csv(file_name,header=head_list)
    
    # now writing the normalized true data
    df_true_norm=pd.DataFrame(x_train,index=train_name)
    file_name = os.path.join(result_folder_fold_path,'train_true_data_norm.csv')
    df_true_norm.to_csv(file_name,header=head_list)
    
    # now writing the true data
    df_true=pd.DataFrame(x_train,index=train_name)
    file_name = os.path.join(result_folder_fold_path,'train_true_data.csv')
    df_true.to_csv(file_name,header=head_list)
    
    # now writing the vector represemtation
    df_rep=pd.DataFrame(train_rep,index=train_name)
    file_name = os.path.join(result_folder_fold_path,'train_vector_rep.csv')
    df_rep.to_csv(file_name,header=[i for i in range(train_rep.shape[1])])
    
    
    
    # now let's begin with the testing data
    x_test_rec_reverse = []
    x_test_original = []
    
    
    for name, temp_rec in zip(test_name,x_test_rec):
        temp_true = data_dict[name]
        temp_true = np.reshape(temp_true,[1,temp_true.shape[0]*temp_true.shape[1]])  
        
        temp_mean = np.mean(temp_true)
        temp_std = np.std(temp_true)
        
        temp_true_norm = (temp_true-temp_mean)/temp_std
        
        x_test_rec_reverse.append(temp_rec*temp_std+temp_mean)    
        x_test_original.append(temp_true)
    
    
    # now writing the recovered data, which have the same scale of the original data
    df_re_data_reverse=pd.DataFrame(x_test_rec_reverse,index=test_name)
    file_name = os.path.join(result_folder_fold_path,'test_recovered_data_reverse.csv')
    df_re_data_reverse.to_csv(file_name,header=head_list)
    
    
    # now writing the recovered data, which is equivalent to the normalized sample 
    df_re_data=pd.DataFrame(x_test_rec,index=test_name)
    file_name = os.path.join(result_folder_fold_path,'test_recovered_data.csv')
    df_re_data.to_csv(file_name,header=head_list)
    
    # now writing the normalized true data
    df_true_norm=pd.DataFrame(x_test,index=test_name)
    file_name = os.path.join(result_folder_fold_path,'test_true_data_norm.csv')
    df_true_norm.to_csv(file_name,header=head_list)
    
    # now writing the true data
    df_true=pd.DataFrame(x_test,index=test_name)
    file_name = os.path.join(result_folder_fold_path,'test_true_data.csv')
    df_true.to_csv(file_name,header=head_list)
    
    # now writing the vector represemtation
    df_rep=pd.DataFrame(test_rep,index=test_name)
    file_name = os.path.join(result_folder_fold_path,'test_vector_rep.csv')
    df_rep.to_csv(file_name,header=[i for i in range(test_rep.shape[1])])
    
    # now drawing the training curves
    
    utils.draw_pic_cost(fold_result,result_folder_fold_path)