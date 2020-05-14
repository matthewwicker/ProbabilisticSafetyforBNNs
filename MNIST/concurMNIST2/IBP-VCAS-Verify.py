import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--eps")
parser.add_argument("--samples")
parser.add_argument("--width")
parser.add_argument("--margin")

args = parser.parse_args()
image = int(args.imnum)
epsilon = float(args.eps)
iters = int(args.samples)
width = int(args.width)
margin = float(args.margin)

import pickle

nproc = 25
import math
import numpy as np
import tensorflow as tf
import ProbablisticReachability
from ProbablisticReachability import interval_bound_propagation_VCAS
from ProbablisticReachability import compute_all_intervals_proc

from my_utils import my_relu
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
x_test = mnist.test.images
y_test = mnist.test.labels

x = x_test[image]
x1 = x
inp = np.asarray(x_test[image])
x_u = np.clip(inp + epsilon, 0, 1)
x_l = np.clip(inp - epsilon, 0, 1)
model_path = "FixedMNIST_Networks/VIMODEL_MNIST_2_%s_relu.net.npz"%(width)

from ProbablisticReachability import gen_samples

try:
    loaded_model = np.load(model_path, allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, mW_2, mb_2, dW_0, db_0, dW_1, db_1, dW_2, db_2] = loaded_model['arr_0']
except:
    with open(model_path, 'r') as pickle_file:
	[mW_0, mb_0, mW_1, mb_1, mW_2, mb_2, dW_0, db_0, dW_1, db_1, dW_2, db_2] = pickle.load(pickle_file)
search_samps = 150
# First, sample and hope some weights satisfy the out_reg constraint
print mW_0.shape, mW_0.shape[0], mW_0.shape[1], type(mW_0)
print dW_0.shape, dW_0.shape[0], dW_0.shape[1], type(dW_0)

sW_0 = np.random.normal(mW_0, dW_0**2, (150, mW_0.shape[0], mW_0.shape[1]))
sb_0 = np.random.normal(mb_0, db_0**2, (150, mb_0.shape[0]))
sW_1 = np.random.normal(mW_1, dW_1**2, (150, mW_1.shape[0], mW_1.shape[1]))
sb_1 = np.random.normal(mb_1, db_1**2, (150, mb_1.shape[0]))
sW_2 = np.random.normal(mW_2, dW_2**2, (150, mW_2.shape[0], mW_2.shape[1]))
sb_2 = np.random.normal(mb_2, db_2**2, (150, mb_2.shape[0]))

y = np.zeros((10))
for i in range(150):
    y += np.matmul(my_relu(np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i]),  sW_2[i]) +sb_2[i]

print "Mean prediction"
print y/150.0

x_reg_1 = [x_l, x_u]
out_cls = np.argmax(y)

ProbablisticReachability.set_model_path(model_path)
ProbablisticReachability.gen_samples(iters)
import time
start = time.time()
from multiprocessing import Pool
p = Pool(nproc)
args = []
for i in range(nproc):
	args.append((x1, x_reg_1, out_cls, margin, iters/nproc, i))
valid_intervals = p.map(interval_bound_propagation_VCAS, args)
p.close()
p.join()

stop = time.time()
elapsed = stop - start
print len(valid_intervals)
log_flag = False
vad_int = []
count_valid = 0
try:
    for i in range(len(valid_intervals)):
        for j in range(len(valid_intervals[i])):
    	    vad_int.append([valid_intervals[i][j][0], 
 			valid_intervals[i][j][1],
			valid_intervals[i][j][2], 
			valid_intervals[i][j][3],
			valid_intervals[i][j][4],
	   		valid_intervals[i][j][5]])
            count_valid +=1
except:
    ph1 = 0.0
    import logging
    logging.basicConfig(filename="Updated_Runs_2_%s.log"%(width),level=logging.DEBUG)
    logging.info("i#=%s_w=%s_e=%s_m=%s_n=%s_t=%s_c=0_p=%s"%(image, width, epsilon, margin, iters, elapsed, ph1))
    log_flag=True

valid_intervals = vad_int
print "IN TOTAL THERE ARE THIS MANY INTERVALS: "
print len(valid_intervals)

#valid_intervals =  interval_bound_propagation_VCAS(x1, x_reg_1, out_cls, w_margin=margin, search_samps=iters)
if(margin != 0):
	p1 = compute_all_intervals_proc((valid_intervals, True, 0, margin, nproc))
	p2 = compute_all_intervals_proc((valid_intervals, False, 1, margin, nproc))
	p3 = compute_all_intervals_proc((valid_intervals, True, 2, margin, nproc))
	p4 = compute_all_intervals_proc((valid_intervals, False, 3, margin, nproc))
	p5 = compute_all_intervals_proc((valid_intervals, True, 4, margin, nproc))
	p6 = compute_all_intervals_proc((valid_intervals, False, 5, margin, nproc))

	ph1 = p1 + p2 + p3 + p4 + p5 + p6
	ph1 = math.exp(ph1)

else:
	ph1 = 0.0
print ph1
stop = time.time()

elapsed = stop - start

import logging
if(not log_flag):
    logging.basicConfig(filename="Updated_Runs_2_%s.log"%(width),level=logging.DEBUG)
    logging.info("i#=%s_w=%s_e=%s_m=%s_n=%s_t=%s_c=%s_p=%s"%(image, width, epsilon, margin, iters, elapsed, len(vad_int), ph1))
