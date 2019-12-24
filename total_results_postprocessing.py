"""
This script is used for summerizing the results after the 5-fold validation is trained.

Output files:
    recovered sequences (output of decoder) from the training samples

    (note that the input of encoder was normalized based on individual instance for data set 2)
    
    vector representation of the training samples    
"""

import numpy as np
import pickle
import os
import utils
import pandas as pd

# first, load the original data
path = 'data_rhythm.csv'

data_dict,_ = utils.file2dict(path,norm=False)

sample_name = list(data_dict.keys())

# then, load the traning results
sub_folder = 'fold_results_h40_bs100_lr0.00100'
result_folder = os.path.join(sub_folder,'total_data')
train_file_path = os.path.join(result_folder,'plotdata.pkl')
train_file = pickle.load(open(train_file_path,'rb'), encoding='latin1')


# recovered data
recover_data = train_file['train_recover_data']
# sample representation
rep = train_file['train_representation']

# got head list
head_list = utils.get_head(path)

recover_data_reverse = []
true_data = []
true_data_norm = []

for name,temp_recover in zip(sample_name,recover_data):
    temp_true = data_dict[name]
    
    temp_true = np.reshape(temp_true,[1,temp_true.shape[0]*temp_true.shape[1]])    
    temp_recover = np.reshape(temp_recover,[1,temp_recover.shape[0]*temp_recover.shape[1]])
    
    temp_mean = np.mean(temp_true)
    temp_std = np.std(temp_true)
    
    temp_true_norm = (temp_true-temp_mean)/temp_std
    
    # the recovered data is multiplied by the 
    # original sample's std and then plus the original sample's mean
    # to make sure they have the same scale
    recover_data_reverse.append(temp_recover*temp_std+temp_mean)
    true_data.append(temp_true)
    
    true_data_norm.append(temp_true_norm)
   
recover_data_reverse = np.squeeze(np.array(recover_data_reverse))
true_data = np.squeeze(np.array(true_data))

# now writing the recovered data, which have the same scale of the original data
df_re_data_reverse=pd.DataFrame(recover_data_reverse,index=sample_name)
file_name = os.path.join(result_folder,'recovered_data_reverse.csv')
df_re_data_reverse.to_csv(file_name,header=head_list)

# now writing the recovered data, which is equivalent to the normalized sample 
recover_data = np.reshape(recover_data,[recover_data.shape[0],recover_data.shape[1]*recover_data.shape[2]])

df_re_data=pd.DataFrame(recover_data,index=sample_name)
file_name = os.path.join(result_folder,'recovered_data.csv')
df_re_data.to_csv(file_name,header=head_list)

# now writing the vector represemtation
df_rep=pd.DataFrame(rep,index=sample_name)
file_name = os.path.join(result_folder,'vector_rep.csv')
df_rep.to_csv(file_name,header=[i for i in range(rep.shape[1])])

true_data_norm = np.squeeze(np.array(true_data_norm))
# now writing the normalized true data
df_true_norm=pd.DataFrame(true_data_norm,index=sample_name)
file_name = os.path.join(result_folder,'true_data_norm.csv')
df_true_norm.to_csv(file_name,header=head_list)

# now drawing the training curves
utils.draw_pic_cost(train_file,result_folder)
