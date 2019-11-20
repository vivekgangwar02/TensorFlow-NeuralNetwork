#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, os, sys, random, warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import import_ipynb
import loaddataset, preprocessing
from playsound import playsound


# In[2]:


class_names = ['banjo', 'cello', 'clarinet', 'french horn', 'guitar', 'oboe', 'trumpet', 'violin']


# In[3]:


def randomfile(path):
    r=re.compile(r'.*_.*4_.*')  #Using regex to take file only from 4th octave
    n=0
    random.seed();
    for root, dirs, files in os.walk(path):
        files=list(filter(r.match,files))
        for name in files:
            n=n+1
            if random.uniform(0, n) < 1:
                rfile=os.path.join(root, name)
    return rfile


# In[4]:


n_input = 50
n_classes = 8
n_hidden_1 = 90

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
def multilayer_perceptron(x, weights, biases, rate):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, rate)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }

biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

rate = tf.placeholder("float")
warnings.filterwarnings('ignore')
predictions = multilayer_perceptron(x, weights, biases, rate)
    
saver = tf.train.Saver()
model_path = "trainedmodel/model.ckpt"

def predict(X):
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Restore model weights from previously saved model
        load_path = saver.restore(sess, model_path)
        warnings.filterwarnings('ignore')

        #Predictions
        pred = predictions.eval(feed_dict = {x:X, rate: 1 })
                                                    #keep_prob is deprecated and will be removed in a future version
        print('This Instrument is',class_names[int(pred.argmax(axis=1))])
    


# In[5]:


def in_notebook(): #To check if code is running in shell or in iPython
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True


# In[8]:


if __name__=="__main__":
    if not in_notebook():
        if len(sys.argv) > 1:
            try:
                path = sys.argv[1]
                playsound(path)
                xtemp = preprocessing.audioToVector(path)
                X = np.array(xtemp)[np.newaxis,:]
                X = (X-min(X[0])) / (max(X[0])-min(X[0]))
                predict(X)
            except Exception:
                print('Something wrong with the File specified')
        else:
            print('''Some random audio will be played and identified automatically:
Press Enter to continue or
Input 'q' to Exit: ''')
            while input()!='q':
                path = randomfile('Dataset/')
                playsound(path)
                xtemp = preprocessing.audioToVector(path)
                X = np.array(xtemp)[np.newaxis,:]
                X = (X-min(X[0])) / (max(X[0])-min(X[0]))
                predict(X)
            print('Exit Command')
    else:
        print('''Some random audio will be played and identified automatically:
Press Enter to continue or
Input 'q' to Exit: ''')
        while input()!='q':
            path = randomfile('Dataset/')
            playsound(path)
            xtemp = preprocessing.audioToVector(path)
            X = np.array(xtemp)[np.newaxis,:]
            X = (X-min(X[0])) / (max(X[0])-min(X[0]))
            predict(X)
        print('Exit Command')


# In[ ]:




