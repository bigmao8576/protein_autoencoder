
import numpy as np
import tensorflow as tf
import pickle
import os
from copy import deepcopy
import utils
import ops
import argparse
import sys

tf.config.optimizer.set_jit(True)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--c_th', type=float,
                        default=0.001,
                        help='the cost threshold for stopping training, default = 0.01')
  
    parser.add_argument('--hidden_num', type=int,
                        default=32,
                        help='the dimension of hidden state, default = 40')
    
    parser.add_argument('--thresh', type=int,
                        default=10,
                        help='threshold for gradient clipping, default = 10')
    parser.add_argument('--batch_size', type=int,
                        default=500,
                        help='batch size, default = 100')

    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='the learning rate, default = 0.0001')
    args = parser.parse_args(argv)
    return args 


def main(argv): 
    args = _parse_args(argv)
    
    cost_threshold = args.c_th
    hidden_num = args.hidden_num
    lr = args.lr
    thresh = args.thresh
    BATCH_SIZE = args.batch_size

    
    

    
    
    plotdata = {"epoch":[],
                "train_cost":[],
                "train_recover_data":[],
                "train_representation":[]
                } 
    
    plotdata['config'] = {
            'cost_threshold':cost_threshold,
            'hidden_num':hidden_num,
            'lr':lr,
            'thresh':thresh,
            'BATCH_SIZE':BATCH_SIZE,
            }
    
    
    # create the folders for saving the results
    first_folder = 'fold_results_h%d_bs%d_lr%0.5f%0.3f'%(hidden_num,BATCH_SIZE,lr,cost_threshold)
    if not os.path.exists(first_folder):
        os.mkdir(first_folder)
    
    plotdata_save_path = os.path.join(first_folder,'total_data')
    if not os.path.exists(plotdata_save_path):
        os.mkdir(plotdata_save_path)
    
    sess_save_path=os.path.join(plotdata_save_path,'save_file')
    if not os.path.exists(sess_save_path):
        os.mkdir(sess_save_path)
    sess_save_path=os.path.join(sess_save_path,'saved_model')
    
    plot_data_path = os.path.join(plotdata_save_path,'plotdata.pkl')    
    if os.path.exists(plot_data_path): # it is continue to train
        cont = 1
        plotdata = pickle.load(open(plot_data_path,'rb'), encoding='latin1')
    else:
        cont = 0
    
    
    
    # import data

    
    data_dict,sample_len = utils.file2dict('data_rhythm.csv')
    x_train,_ = utils.data2array(data_dict)
    
    data_len,max_len = x_train.shape[:2]
    BATCH_NUM = int(np.ceil(len(x_train)/BATCH_SIZE))
    
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None,max_len,2],name='input_signal')
    TH = tf.placeholder(tf.float32,name='grad_th')
    batch_size = tf.placeholder(tf.int64,name='BS')
    
    element,initer,inf_initer = ops.data_iterator(X,batch_size,data_len)
    
    encode_final_state, decode_output = ops.auto_model(element,max_len,hidden_num)
    
    with tf.name_scope('cost'):
        cost=tf.losses.mean_squared_error(
            labels=element,
            predictions=decode_output
        )
    
    
    
    
    opt = tf.train.AdamOptimizer(learning_rate=lr,epsilon=1e-3)
    grads_and_vars = opt.compute_gradients(cost)
    grads, variables = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads,TH)
    
    optimizer = opt.apply_gradients(zip(grads,variables))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        
        if cont:
            i = plotdata["epoch"][-1]+1
            train_cost=plotdata["train_cost"][-1]
            saver.restore(sess, sess_save_path)
        else:
            i=0
            train_cost=1
            
            
        sess.run(initer,feed_dict={X:x_train,batch_size:BATCH_SIZE})
        while train_cost >cost_threshold:
            for _ in range(BATCH_NUM):
                sess.run(optimizer, feed_dict={TH:thresh})
    
            
            if (i) % 100 ==0:
                sess.run(initer,feed_dict={X:x_train,batch_size:len(x_train)})          
                train_cost= sess.run(cost) 
                
                plotdata["epoch"].append(i)
                plotdata["train_cost"].append(train_cost)

    
                
                sess.run(initer,feed_dict={X:x_train,batch_size:BATCH_SIZE})
        
            if (i) % 1000 ==0:
                sess.run(inf_initer,feed_dict={X:x_train,batch_size:len(x_train)})  
                train_rep,train_result_output=sess.run([encode_final_state,decode_output])

                            
    
                plotdata["train_recover_data"]=deepcopy(train_result_output )
                plotdata["train_representation"]=deepcopy(train_rep)

                            
                print('Epoch %d'%i,"--train_cost", train_cost)
    
                utils.draw_pic_cost(plotdata,plotdata_save_path)
                utils.draw_pic_sample(x_train,train_result_output,plotdata_save_path)
                
                
                sess.run(initer,feed_dict={X:x_train,batch_size:BATCH_SIZE})
                saver.save(sess,sess_save_path)      
                pickle.dump(plotdata,open(os.path.join(plotdata_save_path,'plotdata.pkl'),'wb'),protocol=2)
            i=i+1  
        saver.save(sess,sess_save_path)
    plotdata["train_recover_data"]=deepcopy(train_result_output)

    plotdata["train_representation"]=deepcopy(train_rep)
             
    
    pickle.dump(plotdata,open(os.path.join(plotdata_save_path,'plotdata.pkl'),'wb'),protocol=2)
    
if __name__=='__main__': 

    sys.exit(main(sys.argv[1:])) 