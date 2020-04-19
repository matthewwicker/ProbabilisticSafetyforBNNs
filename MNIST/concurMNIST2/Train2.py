
# coding: utf-8

# In[1]:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import edward as ed
from edward.models import Bernoulli, Normal, Categorical,Empirical
from edward.util import Progbar
from keras.layers import Dense
#from scipy.misc import imsave
from edward.util import Progbar
import numpy as np
import gc
from tqdm import trange
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs")
parser.add_argument("--batchsize")
parser.add_argument("--width")
parser.add_argument("--concurID")
parser.add_argument("--activation")

args = parser.parse_args()
set_epochs = int(args.epochs) #remember that this is +2
#set_stepsize = int(args.stepsize)/10000.0 # so pass in 100, 50, 10, 5 to get 0.01, 0.005, 0.001, 0.0005 respectively 
set_width = int(args.width)
batch = int(args.batchsize)
conc = int(args.concurID)
activation = active = activ = str(args.activation)
# In[2]:


# Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
X = mnist.train.images
print X.shape
X = mnist.test.images
print X.shape


# In[3]:


ed.set_seed(980297)
N = batch   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.

inf_steps = 1200000/N # Each one is 10 epochs
width = set_width

# In[4]:


#N = 10  # number of data points
#D = 28 * 28 # number of features

x = tf.placeholder(tf.float32, shape = [N, 784], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("model"):
    
    W_fc1 = Normal(loc=tf.zeros([784, width]), scale=tf.ones([784, width]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([width]), scale=tf.ones([width]), name="b_fc1")
    if(activ == 'relu'):
    	h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    elif(activ == 'sigmoid'):
	h_fc1 = tf.nn.sigmoid(tf.matmul(x, W_fc1) + b_fc1)
    elif(activ == 'tanh'):
	h_fc1 = tf.nn.tanh(tf.matmul(x, W_fc1) + b_fc1)

    W_fc2 = Normal(loc=tf.zeros([width, width]), scale=tf.ones([width, width]), name="W_fc2a")
    b_fc2 = Normal(loc=tf.zeros([width]), scale=tf.ones([width]), name="b_fc2a")

    if(activ == 'relu'):
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    elif(activ == 'sigmoid'):
        h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    elif(activ == 'tanh'):
        h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
    W_fc3 = Normal(loc=tf.zeros([width, 10]), scale=tf.ones([width, 10]), name="W_fc2")
    b_fc3 = Normal(loc=tf.zeros([10]), scale=tf.ones([10]), name="b_fc2")
    #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = Categorical(tf.matmul(h_fc2, W_fc3) + b_fc3)


# In[5]:


# number of samples 
# we set it to 20 because of the memory constrain in the GPU.
# My GPU can take upto about 200 samples at once. 

T = 500
# INFERENCE
post = 2*width
with tf.name_scope("posterior"):
	qW_fc1 = Normal(loc=tf.Variable(tf.random_normal([D, width])),
              scale=1.0/15*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([D, width])))) 
	qb_fc1 = Normal(loc=tf.Variable(tf.random_normal([width])),
              scale=1.0/5*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([width]))))

	qW_fc2 = Normal(loc=tf.Variable(tf.random_normal([width, width])),
              scale=1.0/5*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([width, width])))) 
	qb_fc2 = Normal(loc=tf.Variable(tf.random_normal([width])),
              scale=1.0/5*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([width]))))

	qW_fc3 = Normal(loc=tf.Variable(tf.random_normal([width, K])),
              scale=1.0/5*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([width, K])))) 
	qb_fc3 = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=1.0/5*float(width)*tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# In[6]:


#X_batch , Y_batch = mnist.train.next_batch(N)
#Y_batch = np.argmax(Y_batch, axis = 1)

inference = ed.KLqp({W_fc1: qW_fc1, b_fc1: qb_fc1, 
			W_fc2: qW_fc2, b_fc2: qb_fc2,
			W_fc3: qW_fc3, b_fc3: qb_fc3 }, data={y: y_})
#inference.initialize(step_size=set_stepsize, n_steps=set_steps)
inference.initialize(n_iter=inf_steps, n_print=100, scale={y: float(mnist.train.num_examples) / N})
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[7]:


for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict_hmc = inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
    inference.print_progress(info_dict_hmc)

#print "Lets look at something"
#print qW_fc1.sample().eval().shape
#print qW_fc1.sample(100).eval().shape
#print "DONE"
# In[8]:
# Sample weights and save them to a directory
#train to convergence before this
def save_weights_FAST(epochs_tn):
	import os
	if not os.path.exists("models%s/SampledModels_d=2_w=_%s_e=%s_b=%s_a=%s"%(conc, set_width, epochs_tn, 
											batch, activ)):
    		os.makedirs("models%s/SampledModels_d=2_w=_%s_e=%s_b=%s_a=%s"%(conc, set_width, epochs_tn, 
                                                                                        batch, activ))
	from tqdm import trange
	sW_fc1 = qW_fc1.sample(1000).eval()
	sb_fc1 = qb_fc1.sample(1000).eval()
	sW_fc2 = qW_fc2.sample(1000).eval()
	sb_fc2 = qb_fc2.sample(1000).eval()
	sW_fc3 = qW_fc3.sample(1000).eval()
	sb_fc3 = qb_fc3.sample(1000).eval()
	for _ in trange(1000):
    		np.savez_compressed("models%s/SampledModels_d=2_w=_%s_e=%s_b=%s_a=%s/sampled_weights_%s"%(conc, set_width, epochs_tn, 
                                                                                        batch, activ, _),
						[sW_fc1[_], sb_fc1[_], 
						sW_fc2[_], sb_fc2[_],
						sW_fc3[_], sb_fc3[_]], 
                                                ['wfc1', 'bfc1', 'wfc2', 'bfc2', 'w', 'b'])


#save_weights_FAST(1)


def test_using_last_sample(x_test, y_test):
    x_image = tf.reshape(x_test, [-1,28*28])
    #y_test = np.argmax(y_test, 1).astype("int32")
    W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    #h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)
    W_fc2 = qW_fc2.eval() #.params[-2]
    b_fc2 = qb_fc2.eval() #.params[-2]

    if(activ == 'relu'):
    	h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)
    	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    elif(activ == 'sigmoid'):
	h_fc1 = tf.nn.sigmoid(tf.matmul(x_image, W_fc1) + b_fc1)
	h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
    elif(activ == 'tanh'):
	h_fc1 = tf.nn.tanh(tf.matmul(x_image, W_fc1) + b_fc1)
	h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = qW_fc3.eval() #.params[-2]
    b_fc3 = qb_fc3.eval() #.params[-2]


    y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
    y_pred = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(y_pred , y_test )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )
    return accuracy

print " "
X_test = mnist.test.images
Y_test = mnist.test.labels
Y_test = np.argmax(Y_test,axis=1)
accuracy = test_using_last_sample(X_test ,Y_test)
test_res = accuracy.eval()
print test_res

#save_weights_FAST(1)

# In[15]:


#=============================================
# Iteration 2-n
#=============================================
import gc
import six
from tqdm import tqdm

EPOCHS = set_epochs-1
for i in range(EPOCHS):
    for _ in trange(inference.n_iter):
        X_batch, Y_batch = mnist.train.next_batch(N)
        Y_batch = np.argmax(Y_batch,axis=1)
        info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
        #inference.print_progress(info_dict_hmc)

    accuracy = test_using_last_sample(X_test ,Y_test)
    test_res = accuracy.eval()
    print test_res
    #if((i+1) % 2 == 0 and i != 0):
    #	save_weights_FAST(i+1)
mW_0 = qW_fc1.mean().eval(); dW_0 = qW_fc1.variance().eval()
mb_0 = qb_fc1.mean().eval(); db_0 = qb_fc1.variance().eval()
mW_1 = qW_fc2.mean().eval(); dW_1 = qW_fc2.variance().eval()
mb_1 = qb_fc2.mean().eval(); db_1 = qb_fc2.variance().eval()
mW_2 = qW_fc3.mean().eval(); dW_2 = qW_fc3.variance().eval()
mb_2 = qb_fc3.mean().eval(); db_2 = qb_fc3.variance().eval()

print "Min variance for W0 ", np.min(dW_0), np.mean(dW_0)
print "Min variance for b0 ", np.min(db_0), np.mean(db_0)
print "Min variance for W1 ", np.min(dW_1), np.mean(dW_1)
print "Min variance for b1 ", np.min(db_1), np.mean(db_1)



np.savez_compressed("VIMODEL_MNIST_2_" + str(width) + '_' + activ + ".net", [mW_0, mb_0, mW_1, mb_1, mW_2, mb_2,
                                    dW_0, db_0, dW_1, db_1, dW_2, db_2],
                                   ['mw0', 'mb0', 'mw1', 'mb1',
                                    'dw0', 'db0', 'dw1', 'db1',
				    'dw2', 'db2', 'dw2', 'db2',])

# In[19]:





