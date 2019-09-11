#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:01:32 2019

@author: bigmao
"""
import os
from autoencoder_model import deep_model
import numpy as np
import utils


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"




data_dict,sample_len = utils.file2dict('data_rhythm.csv')

data_fold = utils.data2fold(data_dict)




config ={'MAX_LEN':sample_len,
         'HIDDENT_NUM':16,
         'LAYER_NUM':1,
         'SEQ_CH':2,
         'TH':0.1,
         'INI_LR':1e-3,
         'DECAY_FACTOR':1/10**(1.0/300000)
        }
#model = deep_model(config)

