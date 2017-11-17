import tensorflow as tf
import numpy as np
import time
import sys
from util import *

def weight_variable(shape,sdev=0.1):
    initial = tf.truncated_normal(shape,stddev=sdev,seed=seed)
    return tf.Variable(initial)

def bias_variable(shape,constant=0.1):
    initial = tf.constant(constant,shape=shape)
    return tf.Variable(initial)

def error_measure(prediction,labels):
    return np.sum(np.power(prediction-labels,2))/(2.0*prediction.shape[0])
def eval_in_batches(data,sess,eval_prediction,eval_data_node):
    size = data.shape[0]

    if size <eval_batch_size:
        raise ValueError("batch size for evals larger than dataset siee %d "% size)

    predictions = np.ndarray(shape=(size,num_labels),dtype=np.float32)
    for begin in range(0,size,eval_batch_size):
        end =begin+eval_batch_size
        if end<=size:
            predictions[begin:end,:]=sess.run(eval_prediction,
            feed_dict = {eval_data_node:data[begin:end,...]})

        else:
            batch_predictions = sess.run(eval_prediction,feed_dict={eval_data_node:data[-eval_batch_size:,...]})
            predictions[begin:,:] = batch_predictions[begin-size:,:]

    return predictions

if __name__ =='__main__':
    dprint("loading training set from training csv...")
    train_data,train_label = load('training.csv')
    dprint("loading test set from test.csv")
    test_data,_ =load('test.csv')

    dprint("create validaton data.")
    validation_data = train_data[:validation_size,...]
    validation_labels = train_label[:validation_size]
    train_data =train_data[validation_size:,...]
    train_labels =train_label[validation_size:]

    train_size = train_labels.shape[0]
    dprint("%d training samples." % train_size)

    dprint("constructing tensroflow models...")

    train_data_node = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))
    train_labels_node =tf.placeholder(tf.float32,shape=(batch_size,num_labels))
    eval_data_node = tf.placeholder(tf.float32,shape=(batch_size,image_size,image_size,num_channels))

    conv1_weight= weight_variable([5,5,num_channels,32])
    conv1_biases = bias_variable([32])

    conv2_weight=weight_variable([5,5,32,64])
    conv2_biases =bias_variable([64])

    fc1_weights = weight_variable([image_size//4*image_size//4*64,512])
    fc1_biases  =bias_variable([512])

    fc2_weights = weight_variable([512,512])
    fc2_biases = bias_variable([512])

    fc3_weights = weight_variable([512,num_labels])
    fc3_biases = bias_variable([num_labels])

    def model(data,train =False):
        conv = tf.nn.conv2d(data,conv1_weight,strides=[1,1,1,1],padding='SAME')

        relu =tf.nn.relu(tf.nn.bias_add(conv,conv1_biases))

        pool =tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        conv =tf.nn.conv2d(pool,conv2_weight,strides=[1,1,1,1],padding='SAME')

        #2
        relu = tf.nn.relu(tf.nn.bias_add(conv,conv1_biases))
        pool = tf.nn.max_pool(relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        pool_shape = pool.get_shape().as_list()
        reshape= tf.reshape(pool,[pool_shape[0],pool_shape[1]*pool_shape[2]*pool_shape[3]])


        #fully connected hidden layer
        hidden =tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)

        #training
        if train:
            hidden = tf.nn.dropout(hidden,Dropout_Rate,seed=seed)
        hidden =tf.nn.relu(tf.matmul(hidden,fc1_weights)+fc2_biases)
        return tf.matmul(hidden,fc3_weights)+fc3_biases

    train_prediction = model(train_data_node,True)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction-train_labels_node),1))

    regularizers= (tf.nn.l2_loss(fc1_weights)+tf.nn.l2_loss(fc1_biases)+tf.nn.l2_loss(fc2_weights)+tf.nn.l2_loss(fc2_biases)+tf.nn.l2_loss(fc3_weights)+tf.nn.l2_loss(fc3_biases))

    loss += L2_reg_peram*regularizers

    eval_prediction = model(eval_data_node)

    global_step = tf.Variable(0,trainable=False)

    learning_rate=tf.train.exponential_decay(
        learning_base_rate,
        global_step*batch_size,
        train_size,
        learning_decay_rate,
        staircase=True
    )

    train_step = tf.train.AdamOptimizer(learning_rate,adam_reg_peram).minimize(loss,global_step=global_step)

    dprint("starting tensorflow session...")

    init =tf.initialize_all_variables()
    sess=tf.InteractiveSession()
    sess.run(init)

    loss_train_record = list()
    loss_valid_record = list()

    start_time = time.gmtime()

    best_valid = np.inf
    best_valid_epoch = 0
    current_epoch =0

    dprint("started training session.")

    while current_epoch < num_epochs:
        shuffle_indices = np.arange(train_size)
        np.random.shuffle(shuffle_indices)
        train_data = train_data[shuffle_indices]
        train_labels = train_labels[shuffle_indices]

        for step in range(train_size//batch_size):
            offset =step*batch_size
            batch_data =train_data[offset:(offset+batch_size),...]
            batch_labels = train_labels[offset:(offset+batch_size)]

            feed_dict = {train_data_node:batch_data,
                         train_labels_node:batch_labels}

            _,loss_train, current_learning_rate = sess.run([train_step,loss,learning_rate],feed_dict=feed_dict)

            eval_result = eval_in_batches(validation_data,sess,eval_prediction,eval_data_node)
            loss_valid = error_measure(eval_result,validation_labels)

            dprint("Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f" %
                   (current_epoch, loss_train, loss_valid, loss_train / loss_valid, current_learning_rate))


            loss_train_record.append(np.log10(loss_train))
            loss_valid_record.append(np.log10(loss_valid))

            sys.stdout.flush()

            if loss_valid <best_valid:
                best_valid =loss_valid
                best_valid_epoch = current_epoch

            elif best_valid_epoch+ealry_stop_patience <current_epoch:
                dprint("early stopping")
                dprint ("best valid loss was {:.6f} at epoch{}.".format(best_valid,best_valid_epoch))
                break

            current_epoch+=1

        dprint("training finished")
        end_time = time.gmtime()

        dprint(time.strftime('%H:%H:%S',start_time))
        dprint(time.strftime('%H:%H:%S',end_time))

        dprint("Generating test result.")
        generate_test_results(test_data,sess,eval_prediction,eval_data_node)