import tensorflow as tf

def data_iterator(X,batch_size,data_len):
    
    '''
    input:
        X: a placeholder, the input data, can be either training or testing data
        batch_size: a placeholder
        data_len: the number of samples
    output:
        element: the tf.dataset element
        initer: initializer for the training iterater
        inf_initer: the initializer for the testing iterater
    '''
    
    ds = tf.data.Dataset.from_tensor_slices((X))
    ds = ds.shuffle(data_len)
    ds = ds.batch(batch_size)
    ds = ds.repeat()
    input_data = ds.prefetch(buffer_size=1000)
        
    inf = tf.data.Dataset.from_tensor_slices((X))
    inf = inf.batch(batch_size)
        
    Iterator=tf.data.Iterator.from_structure(input_data.output_types,input_data.output_shapes)        
    initer= Iterator.make_initializer(input_data)
    inf_initer= Iterator.make_initializer(inf)
    
    
    element = Iterator.get_next()
    
    return element,initer,inf_initer

def auto_model(element,max_len,hidden_num):
    
    Y = tf.zeros([tf.shape(element)[0],max_len,hidden_num])
    
    encode_cell_fw = tf.keras.layers.GRU(hidden_num//2)
    encode_cell_bw = tf.keras.layers.GRU(hidden_num//2,go_backwards=True)
    
    
    encode_final_state = tf.keras.layers.Bidirectional(encode_cell_fw, backward_layer=encode_cell_bw)(element)
    
    decode_cell = tf.keras.layers.GRU(hidden_num,return_sequences=True)
    decode_seq = decode_cell(inputs = Y, initial_state = encode_final_state)
    
    decode_output=tf.contrib.layers.fully_connected(decode_seq,num_outputs=2,activation_fn=None)
    
    return encode_final_state, decode_output