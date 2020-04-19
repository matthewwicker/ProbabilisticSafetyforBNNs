"""
REQUIREMENTS FOR THIS FILE:
Python 3 Interpreter

NUMPY 	----	1.18.1
H5PY	----	2.10.0
KERAS	----	2.3.1
TENSORFLOW -	1.14.0
"""

import sys
import h5py
import math
import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adamax, Nadam, RMSprop, Adam
import tensorflow as tf

# Check your version compliance below.
#print(np.__version__)
#print(h5py.__version__)
#print(keras.__version__)
#print(tf.__version__)

np.random.seed(1)
tf.random.set_random_seed(1)

# We will use an interactive session for the final posterior save.
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

######## OPTIONS #########
ver = 4            # Neural network version - This is from VCAS devs
trainingDataFiles = "../TrainingData/VertCAS_TrainingData_v2_%02d.h5" # File format for training data
pra = 1 # We assume that we currently have COC - Clear of Conflict (for simplicity)
print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
f       = h5py.File(trainingDataFiles % pra,'r')
X_train = np.array(f['X'])
Q       = np.array(f['y'])
means = np.array(f['means'])
ranges=np.array(f['ranges'])
min_inputs = np.array(f['min_inputs'])
max_inputs = np.array(f['max_inputs'])
                       
N,numOut = Q.shape

##########################

####### CUSTOM KERAS BAYES LAYER ########
####### CREDIT FOR THIS CODE TO: ########
#######      Martin Krasser      ########
#krasserm.github.io/2019/03/14/bayesian-neural-networks/

from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
from keras.layers import Dense
import tensorflow as tf
import tensorflow_probability as tfp

class DenseVariational(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.0,
                 prior_sigma_2=1.0,
                 prior_pi=0.0, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.normal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.normal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))


    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))


############################
from sklearn.metrics import confusion_matrix
#from: https://gist.github.com/zachguo/10296432
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Classwise accuracies:")
    print(cm.diagonal())
    """pretty print for confusion matrices"""
    columnwidth = 9 #max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

######################

print("Setting up Model")
    
# Asymmetric loss function
lossFactor = 10000.0
def asymMSE(y_true, y_pred):
    d = y_true-y_pred
    maxes = tf.argmax(y_true,axis=1)
    maxes_onehot = tf.one_hot(maxes,numOut)
    others_onehot = maxes_onehot-1
    d_opt = d*maxes_onehot 
    d_sub = d*others_onehot
    a = lossFactor*(numOut-1)*(tf.square(d_opt)+tf.abs(d_opt))
    b = tf.square(d_opt)
    c = lossFactor*(tf.square(d_sub)+tf.abs(d_sub))
    d = tf.square(d_sub)
    loss = tf.where(d_sub>0,c,d) + tf.where(d_opt>0,a,b)
    return tf.reduce_mean(loss)

trunc_data, trunc_labs = [],[]
for i in range(len(X_train)):
    if(np.argmax(Q[i]) == 0):
        continue
    else:
        if(np.argmax(Q[i]) == 1):
            print(X_train[i])
        trunc_data.append(X_train[i])
        trunc_labs.append(Q[i])
trunc_data, trunc_labs = np.asarray(trunc_data), np.asarray(trunc_labs)
# Define model architecture
num_batches = 1000
kl_weight = 100.0 / num_batches

model = Sequential()
model.add(DenseVariational(512, kl_weight, activation='relu', input_shape=(4,)))
model.add(DenseVariational(numOut, kl_weight=0.5))
opt = Nadam(lr = 0.005)

labels = range(1,5)
model.compile(loss=asymMSE, optimizer=opt, metrics=['accuracy', 'mse'])

# Train and write nnet files
model.fit(trunc_data, trunc_labs, nb_epoch=20, batch_size=512, shuffle=True)
y_pred = model.predict(trunc_data, verbose=1, batch_size=8192)
y_pred = np.argmax(y_pred, axis=1);
y = np.argmax(trunc_labs, axis=1)
cm = confusion_matrix(y_pred, y, labels)
print_cm(cm, labels)


y_pred = model.predict(trunc_data, verbose=1, batch_size=8192)
y_pred = np.argmax(y_pred, axis=1);
y = np.argmax(trunc_labs, axis=1)
cm = confusion_matrix(y_pred, y, labels)
print_cm(cm, labels)

# Now we need to save the Bayesian posterior
mW_0 = model.layers[0].kernel_mu.eval()
dW_0 = tf.math.softplus(model.layers[0].kernel_rho).eval()
mb_0 = model.layers[0].bias_mu.eval()
db_0 = tf.math.softplus(model.layers[0].bias_rho).eval()
mW_1 = model.layers[1].kernel_mu.eval()
dW_1 = tf.math.softplus(model.layers[1].kernel_rho).eval()
mb_1 = model.layers[1].bias_mu.eval()
db_1 = tf.math.softplus(model.layers[1].bias_rho).eval()
width = 512

import pickle
with open( "VCAS_MODEL_" + str(width) + ".net", 'wb') as pickle_file:
	pickle.dump([mW_0, mb_0, mW_1, mb_1,
                     dW_0, db_0, dW_1, db_1], pickle_file, protocol=2)
