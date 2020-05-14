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

nproc = 50
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
model_path = "FixedMNIST_Networks/VIMODEL_MNIST_1_%s_relu.net.npz"%(width)

from ProbablisticReachability import gen_samples
try:
    loaded_model = np.load(model_path, allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
except:
    with open(model_path, 'rb') as pickle_file:
	[mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
search_samps = 150
# First, sample and hope some weights satisfy the out_reg constraint
sW_0 = np.random.normal(mW_0, dW_0**2, (150, mW_0.shape[0], mW_0.shape[1]))
sb_0 = np.random.normal(mb_0, db_0**2, (150, mb_0.shape[0]))
sW_1 = np.random.normal(mW_1, dW_1**2, (150, mW_1.shape[0], mW_1.shape[1]))
sb_1 = np.random.normal(mb_1, db_1**2, (150, mb_1.shape[0]))

y = np.zeros((10))
for i in range(150):
    y += (np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])

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
if(len(valid_intervals) == 0):
    import logging
    ph1 = 0.0
    logging.basicConfig(filename="Updated_Runs%s.log"%(width),level=logging.DEBUG)
    logging.info("i#=%s_w=%s_e=%s_m=%s_n=%s_t=%s_p=%s"%(image, width, epsilon, margin, iters, elapsed, ph1))



vad_int = []
vW_0 = []
vb_0 = []
vW_1 = []
vb_1 = []
logged_flag = False
try:
    for i in range(len(valid_intervals)):
        for j in range(len(valid_intervals[i])):
    	    vad_int.append([valid_intervals[i][j][0], 
 			valid_intervals[i][j][1],
			 valid_intervals[i][j][2], 
			valid_intervals[i][j][3]])
except:
    import logging
    ph1 = 0.0
    logging.basicConfig(filename="Updated_Runs%s.log"%(width),level=logging.DEBUG)
    logging.info("i#=%s_w=%s_e=%s_m=%s_n=%s_t=%s_c=0_p=%s"%(image, width, epsilon, margin, iters, elapsed, ph1))
    logged_flag = True
valid_intervals = vad_int
print "IN TOTAL THERE ARE THIS MANY INTERVALS: "
print len(valid_intervals)
"""
pW_0 = ProbablisticReachability.compute_interval_probs_weight(np.asarray(vW_0), marg=margin, mean=mW_0, std=dW_0)
pb_0 = ProbablisticReachability.compute_interval_probs_bias(np.asarray(vb_0), marg=margin, mean=mb_0, std=db_0)
pW_1 = ProbablisticReachability.compute_interval_probs_weight(np.asarray(vW_1), marg=margin, mean=mW_1, std=dW_1)
pb_1 = ProbablisticReachability.compute_interval_probs_bias(np.asarray(vb_1), marg=margin, mean=mb_1, std=db_1)


p = 0.0
for i in pW_0.flatten():
    p+=math.log(i)
for i in pb_0.flatten():
    p+=math.log(i)
for i in pW_1.flatten():
    p+=math.log(i)
for i in pb_1.flatten():
    p+=math.log(i)
p = math.exp(p)
ph1 = p
"""
#valid_intervals =  interval_bound_propagation_VCAS(x1, x_reg_1, out_cls, w_margin=margin, search_samps=iters)
if(margin != 0):
	p1 = compute_all_intervals_proc((valid_intervals, True, 0, margin, nproc))
	p2 = compute_all_intervals_proc((valid_intervals, False, 1, margin, nproc))
	p3 = compute_all_intervals_proc((valid_intervals, True, 2, margin, nproc))
	p4 = compute_all_intervals_proc((valid_intervals, False, 3, margin, nproc))

	ph1 = p1 + p2 + p3 + p4
	ph1 = math.exp(ph1)

else:
	ph1 = 0.0
stop = time.time()
print ph1
print ph1
print ph1

elapsed = stop - start
if(not logged_flag):
    import logging
    logging.basicConfig(filename="Updated_Runs%s.log"%(width),level=logging.DEBUG)
    logging.info("i#=%s_w=%s_e=%s_m=%s_n=%s_t=%s_c=%s_p=%s"%(image, width, epsilon, margin, iters, elapsed, len(valid_intervals), ph1))
