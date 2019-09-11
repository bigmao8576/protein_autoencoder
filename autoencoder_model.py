#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:41:19 2019

@author: bigmao
"""

import tensorflow as tf
import os

class deep_model:
    def __init__(self,config):
        '''
        config: The configuration if model.
            'MAX_LEN'        --- The maximum length of the sequence
            'HIDDENT_NUM'    --- The hidden size
            'LAYER_NUM'      --- The layer number
            'SEQ_CH'         --- The channel number (dimension) of input sequence
            'TH'             --- threshold for clipping gradient
            'INI_LR'         --- The initial learning rate
            'DECAY_FACTOR'   --- For learning rate decay
        '''            
        tf.reset_default_graph()
        

        
        self.cpu_cores = os.cpu_count()//2
        self.config = config
        
        self.X = tf.placeholder(tf.float32, [None,self.config['MAX_LEN'],self.config['SEQ_CH']],name='input_signal')
        self.Y = tf.zeros([tf.shape(self.X)[0],self.config['MAX_LEN'],self.config['HIDDENT_NUM']], name = 'Y_in')
        self.L = tf.placeholder(tf.int32, [None],name='sample_length')
        self.TH = tf.placeholder(tf.float32,name='grad_th')
        self.lr = tf.placeholder(tf.float32,name='learning_rate')
        
        sys_config = tf.ConfigProto(allow_soft_placement=True)
        sys_config.gpu_options.allow_growth = True                
        self.sess = tf.Session(config=sys_config)
        
        self.model()
        self.loss()
        self.opt()
        
        self.globel_lr = config.ini_lr
        
        self.sess.run(tf.global_variables_initializer())
    def model(self):
        encode_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.GRUCell(self.config['HIDDENT_NUM'], reuse=False)
              for _ in range(self.config['LAYER_NUM'])]
        )
        encode_output, self.encode_final_state = tf.nn.dynamic_rnn(
        encode_cell, self.X, self.L, dtype=tf.float32, scope='encode')
        
        decode_cell = tf.contrib.rnn.MultiRNNCell(
          [tf.contrib.rnn.GRUCell(self.config['HIDDENT_NUM'], reuse=False)
              for _ in range(self.config['LAYER_NUM'])]
        )
        decode_cell_output, _ = tf.nn.dynamic_rnn(decode_cell, self.Y, sequence_length=self.L, initial_state=self.encode_final_state, scope='decode')
        self.decode_output=tf.contrib.layers.fully_connected(decode_cell_output,num_outputs=self.config['SEQ_CH'],activation_fn=None)
        
    def loss(self):
        self.cost=tf.losses.mean_squared_error(
            labels=tf.reverse(self.X,[1]),
            predictions=self.decode_output
        )
        
    def opt(self):
            
        opt = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=1e-3)
        grads_and_vars = opt.compute_gradients(self.cost)
        grads, variables = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads,self.TH)
        
        self.optimizer = opt.apply_gradients(zip(grads,variables))
        self.norms = tf.global_norm(grads)
        
    def train(self,x_train,L_train):
        
        _, grads_norm= self.sess.run([self.optimizer,self.norms], 
                                     feed_dict={self.X:x_train,
                                                self.L:L_train,
                                                self.lr:self.globel_lr, 
                                                self.TH:self.config['TH']})
        self.globel_lr *= self.config['DECAY_FACTOR']
        
    def evaluation(self,x_train,L_train,x_test,L_test):
        train_cost= self.sess.run(self.cost, feed_dict={self.X:x_train,self.L:L_train})   
        test_cost= self.sess.run(self.cost, feed_dict={self.X:x_test,self.L:L_test}) 
        
        return train_cost,test_cost
    
    def apply(self,x_train,L_train,x_test,L_test):
        train_rep,train_result_output=self.sess.run([self.encode_final_state,self.decode_output],feed_dict={self.X:x_train,self.L:L_train})
        test_rep,test_result_output=self.sess.run([self.encode_final_state,self.decode_output],feed_dict={self.X:x_test,self.L:L_test})
        
        train_result_output = train_result_output[:,::-1,:]
        test_result_output = test_result_output[:,::-1,:]
        
    def save_model(self,sess_save_path):
        
        if not os.path.exists(sess_save_path):
            os.makedirs(sess_save_path)
            
        sess_save_path=os.path.join(sess_save_path,'model_save')
        self.saver.save(self.sess,sess_save_path)
        
    def load_model(self,sess_save_path):
        self.saver.restore(self.sess,sess_save_path)