import numpy as np
import pickle
#from tqdm import trange
import edward as ed
import tensorflow as tf
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

model_path = "ERR - NO MODEL SET. Call set_model_path function."
def set_model_path(m):
    global model_path
    model_path = m

GLOBAL_samples = "lollol"
def gen_samples(iters):
    global GLOBAL_samples
    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    except:
        with open(model_path, 'r') as pickle_file:
            [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
    sW_0 = np.random.normal(mW_0, dW_0, (iters, mW_0.shape[0], mW_0.shape[1]))
    sb_0 = np.random.normal(mb_0, db_0, (iters, mb_0.shape[0]))
    sW_1 = np.random.normal(mW_1, dW_1, (iters, mW_1.shape[0], mW_1.shape[1]))
    sb_1 = np.random.normal(mb_1, db_1, (iters, mb_1.shape[0]))
    GLOBAL_samples = [sW_0, sb_0, sW_1, sb_1]
     

"""
Interval propogation in weight and input space. 
@Variable W - the sampled weight that gave an output in the valid output region
@Variable qW - the variational posterior weight matrix values (mean and variance)
@Variable b - the sampled bias that gave an output in the valid output region
@Variable qb - the variational posterior bias vector values (mean and variance)
@Variable x_l - the lower bound of the input region
@Variable x_u - the upper bound of the input region
@Variable eps - the margin to propagate in the weight space (we add and subtract this value)
"""
def propagate_interval(W, W_std, b, b_std, x_l, x_u, eps):
    W_l, W_u = W-(eps*W_std), W+(eps*W_std)    #Use eps as small symetric difference about the mean
    b_l, b_u = b-(eps*b_std), b+(eps*b_std)   #Use eps as small symetric difference about the mean 
    h_max = np.zeros(len(W[0])) #Placeholder variable for return value
    h_min = np.zeros(len(W[0])) #Placeholder variable for return value
    for i in range(len(W)):     #This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])): # where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         #Return the min and max of the intervals.
                                #(dont forget to apply activation function after)

# Code for merging overlapping intervals. Taken from here: 
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for  i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1] 

"""
Given a set of disjoint intervals, compute the probability of a random
sample from a guassian falling in these intervals. (Taken from lemma)
of the document
"""
import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, stddev):
    prob = 0.0
    for interval in intervals:
        val1 = erf((mean-interval[0])/(math.sqrt(2)*(stddev)))
        val2 = erf((mean-interval[1])/(math.sqrt(2)*(stddev)))
        prob += 0.5*(val1-val2)
    #if(prob < 0.99):
    #    print intervals
    #    print mean
    #    print stddev
    return prob

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a weight matrix
"""
def compute_interval_probs_weight(vector_intervals, marg, mean, std):
    means = mean; stds = std
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(stds[i][j]*marg), vector_intervals[num_found][i][j]+(stds[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], stds[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a *flat* bias matrix (vector)
"""
def compute_interval_probs_bias(vector_intervals, marg, mean, std):
    means = mean; stds = std
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(stds[i]*marg), vector_intervals[num_found][i]+(std[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], stds[i])
        prob_vec[i] = p
    return np.asarray(prob_vec)

def compute_single_prob(arg):
    vector_intervals, marg, mean, std = arg
    intervals = []
    for num_found in range(len(vector_intervals)):
        #!*! Need to correct and make sure you scale margin
        interval = [vector_intervals[num_found]-(std*marg), vector_intervals[num_found]+(std*marg)]
        intervals.append(interval)
    p = compute_erf_prob(merge_intervals(intervals), mean, std)
    if(p == 0.0):
	print "error"
	return float("-inf")
    return p
"""
Probabalistic Reachability for Bayesian Neural Networks - V 0.0.1 - Variable Margin
@Variable x - the original input (not used, may delete)
@Variable in_reg - a list [x_l, x_u] containing the region of interest in the input space
@Variable out_reg - a list [y_l, y_u] containing the region of interest in the output space
@Variable w_margin - a float value dictating the amount to add and subtract to create compact 
                     set in the weight space given some valid sample from weight space
@Variable search_samps - number of posterior samples to take in order to check for valid
                         samples (i.e. samples that cause output to be in valid range)
                         
@Return - A valid lowerbound on the probability that the input region causes BNN  to give
          ouput bounded by the output region. Converges to exact solution when margin is
          small and samples goes to infinity.
"""

from my_utils import my_relu
import pickle


def interval_bound_propagation_VCAS(a):
    x, in_reg, out_maximal, w_margin, search_samps, id = a
    act = 'relu'
    reverse = False
    x = np.asarray(x); x = x.astype('float64')
    x_l, x_u = in_reg[0], in_reg[1]
    out_ind = out_maximal
    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    except:
        with open(model_path, 'r') as pickle_file:
	    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
    # First, sample and hope some weights satisfy the out_reg constraint
    [sW_0, sb_0, sW_1, sb_1] = GLOBAL_samples
    print id*search_samps,(id+1)*search_samps
    sW_0 = sW_0[id*search_samps:(id+1)*search_samps]
    sb_0 = sb_0[id*search_samps:(id+1)*search_samps]
    sW_1 = sW_1[id*search_samps:(id+1)*search_samps]
    sb_1 = sb_1[id*search_samps:(id+1)*search_samps]
    valid_weight_intervals = []
    err = 0
    for i in range(search_samps): 
        # Full forward pass in one line :-)
        if act == 'tanh':
            y = pface*(tf.matmul(tf.tanh(tf.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i]).eval()
        elif act == 'relu':
            y = (np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])
        # Logical check if weights sample out_reg constraint
        #print sW_0[i][0][0]
        extra_gate = (reverse and np.argmax(y) != out_ind)
        if(np.argmax(y) == out_ind or extra_gate):
            # If so, do interval propagation
            h_l, h_u = propagate_interval(sW_0[i], dW_0, sb_0[i], db_0, x_l, x_u, w_margin)
            if act == 'tanh':
                h_l, h_u = np.tanh(h_l), np.tanh(h_u)
            elif act == 'relu':
                h_l, h_u = my_relu(h_l), my_relu(h_u)
            y_pred_l, y_pred_u = propagate_interval(sW_1[i], dW_1, sb_1[i], db_1, h_l, h_u, w_margin)
            assert((y_pred_l <= y).all())
            assert((y_pred_u >= y).all())
            # Check if interval propagation still respects out_reg constraint
            #extra_gate = (reverse and np.argmax(y_pred_l) != out_ind and np.argmax(y_pred_u) != out_ind)
            safety_check = True; value_ind = 0
            for value in y_pred_u:
                if(y_pred_l[out_ind] < value and value_ind != out_ind):
                    safety_check = False
                value_ind += 1
	    if(safety_check):
                # If it does, add the weight to the set of valid weights
                valid_weight_intervals.append([sW_0[i], sb_0[i], sW_1[i], sb_1[i]])
        else:
	    err += 1
	    #print np.argmax(y)
            #print y
	    #print "Hm, incorrect prediction is worrying..."
            continue
    print "We found %s many valid intervals."%(len(valid_weight_intervals))
    print "Pred error rate: %s/%s"%(err/float(search_samps), err)
    if(len(valid_weight_intervals) == 0):
        return 0.0
    # Now we need to take all of the valid weight intervals we found and merge them
    # so we seperate the valid intervals into their respective variables
    """
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    for v in valid_weight_intervals:
        #np.asarray(v[0]) i removed this... should i not have? -MW
        vW_0.append(v[0])
        vb_0.append(v[1])
        vW_1.append(v[2])
        vb_1.append(v[3])
    """
    return valid_weight_intervals

    """
     everything after this is now not running
    """
    # After we merge them, we need to use the erf function to evaluate exactly what the 
    #   lower bound on the probability is!
    pW_0 = compute_interval_probs_weight(np.asarray(vW_0), marg=w_margin, mean=mW_0, std=dW_0)
    pb_0 = compute_interval_probs_bias(np.asarray(vb_0), marg=w_margin, mean=mb_0, std=db_0)
    pW_1 = compute_interval_probs_weight(np.asarray(vW_1), marg=w_margin, mean=mW_1, std=dW_1)
    pb_1 = compute_interval_probs_bias(np.asarray(vb_1), marg=w_margin, mean=mb_1, std=db_1)
    
    # Now that we have all of the probabilities we just need to multiply them out to get
    # the final lower bound on the probability of the condition holding.
    # Work with these probabilities in log space
    p = 0.0
    for i in pW_0.flatten():
        p+=math.log(i)
    for i in pb_0.flatten():
        p+=math.log(i)
    for i in pW_1.flatten():
        p+=math.log(i)
    for i in pb_1.flatten():
        p+=math.log(i)
    #print math.exp(p)
    return math.exp(p)
    
    
from multiprocessing import Pool
def compute_all_intervals_proc(a):
    V, isweight, i, margin, numproc = a
    print("STARTING COMPUTE FOR %s, %s"%(i,isweight))

    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    except:
        with open(model_path, 'r') as pickle_file:
            [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
    vW_0 = []
    for valid_intervals in V:
        vW_0.append(valid_intervals[i])

    vW_0 = np.asarray(vW_0)

    if(isweight):
        vW_0 = np.swapaxes(vW_0, 0, 2)
        vW_0 = np.swapaxes(vW_0, 0, 1)
    else:
        vW_0 = np.swapaxes(vW_0, 0, 1)
    print vW_0.shape
    # After we merge them, we need to use the erf function to evaluate exactly what the 
    #   lower bound on the probability is!
    ind = i
    nproc = numproc
    print "Using %s processes"%(nproc)
    p = Pool(nproc)
    
    # Need something more general here for the multilayer case
    if(ind == 0):

        print vW_0.shape
        pW_0 = np.ones((vW_0.shape[0], vW_0.shape[1]))
        for i in range(len(vW_0)):
            args = []
            for j in range(len(vW_0[i])):
                args.append((vW_0[i][j], margin, mW_0[i][j], dW_0[i][j]))
            arr = p.map(compute_single_prob, args)
            pW_0[i] = arr
    elif(ind == 2):
        pW_0 = np.ones((vW_0.shape[0], vW_0.shape[1]))
        for i in range(len(vW_0)):
            args = []
            for j in range(len(vW_0[i])):
                args.append((vW_0[i][j], margin, mW_1[i][j], dW_1[i][j]))
            arr = p.map(compute_single_prob, args)
            pW_0[i] = arr
    elif(ind == 1):
        pW_0 = np.ones((vW_0.shape[0]))
        args = []
        for i in range(len(vW_0)):
            args.append((vW_0[i], margin, mb_0[i], db_0[i]))
        pW_0 = p.map(compute_single_prob, args)

    elif(ind == 3):
        pW_0 = np.ones((vW_0.shape[0]))
        args = []
        for i in range(len(vW_0)):
            args.append((vW_0[i], margin, mb_1[i], db_1[i]))
        pW_0 = p.map(compute_single_prob, args)
    pW_0 = np.asarray(pW_0)
    print "HERE IS THE SHAPE"
    print pW_0.shape
    p.close()
    p.join()
    p = 0.0
    for i in pW_0.flatten():
        try:
            p+=math.log(i)
        except:
            return float("-inf")
    return p



# !! THIS IS ONLY FOR RELU !! NO OTHER ACTIVATION SUPPORTED atm
def get_alphas_betas(zeta_l, zeta_u, activation="relu"):
    alpha_L, alpha_U = list([]), list([])
    beta_L, beta_U = list([]), list([])
    for i in range(len(zeta_l)):
        if(zeta_u[i] <= 0):
            alpha_U.append(0); alpha_L.append(0); beta_L.append(0); beta_U.append(0)
        elif(zeta_l[i] >= 0):
            alpha_U.append(1); alpha_L.append(1); beta_L.append(0); beta_U.append(0)
        else:
            # For relu I have the points (zeta_l, 0) and (zeta_u, zeta_u)
            a_U = zeta_u[i]/(zeta_u[i]-zeta_l[i]); b_U = -1*(a_U*zeta_l[i])
    
            #a_L = a_U ; b_L = 0
            #if (zeta_u[i] + zeta_l[i]) >= 0:
            #    a_L = 1 ;   b_L = 0
            #else:
            a_L = 0 ;   b_L = 0    
            alpha_U.append(a_U); alpha_L.append(a_L); beta_L.append(b_L); beta_U.append(b_U)
    return alpha_U, beta_U, alpha_L, beta_L



def get_bar_lower(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l); mu_u = np.squeeze(mu_u); 
    mu_bar, nu_bar, lam_bar = [], [], []
    
    nu_bar = nu_l

    #coef of the form - alpha_U, beta_U, alpha_L, beta_L
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,2] >= 0):
            mu_bar.append(linear_bound_coef[i,2] * mu_l[i])
            for k in range(len(nu_bar)):
                try:
                    nu_bar[k][i] = linear_bound_coef[i,2] * np.asarray(nu_l[k][i])
                except:
                    print 'error'
            lam_bar.append(linear_bound_coef[i,2] * lam_l[i] + linear_bound_coef[i,3])
        else:
            mu_bar.append(linear_bound_coef[i,2] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,2] * nu_u[k][i]
            lam_bar.append(linear_bound_coef[i,2] * lam_u[i] + linear_bound_coef[i,3])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)

def get_bar_upper(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l); mu_u = np.squeeze(mu_u);  
    mu_bar, nu_bar, lam_bar = [], [], []
    nu_bar = nu_u
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,0] >= 0):
            mu_bar.append(linear_bound_coef[i,0] * mu_u[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,0] * np.asarray(nu_u[k][i])
            lam_bar.append(linear_bound_coef[i,0] * lam_u[i] + linear_bound_coef[i,1])
        else:
            mu_bar.append(linear_bound_coef[i,0] * mu_l[i])
            for k in range(len(nu_bar)):
                nu_bar[k][i] = linear_bound_coef[i,0] * nu_l[k][i]
            lam_bar.append(linear_bound_coef[i,0] * lam_l[i] + linear_bound_coef[i,1])
    return np.asarray(mu_bar), nu_bar, np.asarray(lam_bar)

def get_abc_lower(w, mu_l_bar, nu_l_bar, la_l_bar,
               mu_u_bar, nu_u_bar, la_u_bar):
    a, b, c = [], [], []
    for i in range(len(w)):
        curr_a = []
        #curr_b = []
        curr_c = []
        for j in range(len(w[i])):
            if(w[i][j] >= 0):
                curr_a.append(w[i][j] * mu_l_bar[i])
                curr_c.append(w[i][j] * la_l_bar[i])
            else:
                curr_a.append(w[i][j] * mu_u_bar[i])
                curr_c.append(w[i][j] * la_u_bar[i])
        a.append(curr_a)
        
        c.append(curr_c)
    for k in range(len(nu_l_bar)): 
        curr_b = []
        #for i in range(len(w)):
        for j in range(len(w[i])):
            curr_curr_b = []
            #for j in range(len(w[i])):
            for i in range(len(w)):
                if(w[i][j] >= 0):
                    curr_curr_b.append(w[i][j] * nu_l_bar[k][i])
                else:
                    curr_curr_b.append(w[i][j] * nu_u_bar[k][i])
            curr_b.append(curr_curr_b)
        b.append(curr_b)  
        
        
    return np.asarray(a), b, np.asarray(c)


def get_abc_upper(w, mu_l_bar, nu_l_bar, la_l_bar,
               mu_u_bar, nu_u_bar, la_u_bar):
    #This is anarchy
    return get_abc_lower(w,mu_u_bar, nu_u_bar, la_u_bar,
                         mu_l_bar, nu_l_bar, la_l_bar)


def min_of_linear_fun(coef_vec, uppers, lowers):
   #getting the minimum
    val_min = 0
    for i in range(len(coef_vec)):
        if coef_vec[i] >=0:
            val_min = val_min + coef_vec[i]*lowers[i]
        else: 
            val_min = val_min + coef_vec[i]*uppers[i]
    return val_min

def max_of_linear_fun(coef_vec, uppers, lowers):
    val_max = - min_of_linear_fun(-coef_vec, uppers, lowers)
    return val_max


def propogate_lines(x, in_reg, sWs,sbs,
                    w_margin=0.25, search_samps=100, act = 'relu'):

    x = np.asarray(x); x = x.astype('float64')
    x_l, x_u = in_reg[0], in_reg[1]
    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
        dWs = [dW_0,dW_1]
        dbs = [db_0,db_1]
        widths = [512]   
    except:
        with open(model_path, 'r') as pickle_file:
            [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
            dWs = [dW_0,dW_1]
            dbs = [db_0,db_1]
            widths = [512]   
    
    #Code compatibility - Going from the multivariable format to the list format        
    #sW_0 = np.random.normal(mW_0, dW_0, (search_samps, mW_0.shape[0], mW_0.shape[1]))
    #sb_0 = np.random.normal(mb_0, db_0, (search_samps, mb_0.shape[0]))
    #sW_1 = np.random.normal(mW_1, dW_1, (search_samps, mW_1.shape[0], mW_1.shape[1]))
    #sb_1 = np.random.normal(mb_1, db_1, (search_samps, mb_1.shape[0]))
     
    
    n_hidden_layers = len(widths)
    
    #sWs = [sW_0,sW_1]
    #sbs = [sb_0,sb_1]
    
    
    #Code adaptation end. From now on it's the standard code 

        
        
    #Actual code from now on    
    
    #Step 1: Inputn layers -> Pre-activation function        
    W_0_L, W_0_U, b_0_L, b_0_U = (sWs[0][0] - dWs[0]*w_margin,  sWs[0][0] + dWs[0]*w_margin, 
                                  sbs[0][0]-dbs[0]*w_margin, sbs[0][0]+dbs[0]*w_margin)
    
    W_0_L = W_0_L.T
    W_0_U = W_0_U.T
    
    mu_0_L = W_0_L; mu_0_U = W_0_U
    
    n_hidden_1 = sWs[0][0].shape[1]
    
    nu_0_L = np.asarray([x_l for i in range(n_hidden_1) ])
    nu_0_U = np.asarray([x_l for i in range(n_hidden_1) ])
    la_0_L = - np.dot(x_l, W_0_L.T) + b_0_L
    la_0_U = - np.dot(x_l, W_0_U.T) + b_0_U
    
    
    # getting bounds on pre-activation fucntion
    zeta_0_L = [ (min_of_linear_fun(np.concatenate((mu_0_L[i].flatten(), nu_0_L[i].flatten())), 
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten() )),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten() ))  )) for i in range(n_hidden_1)] 
   
    zeta_0_L = np.asarray(zeta_0_L) + la_0_L
     
    zeta_0_U = [ (max_of_linear_fun(np.concatenate((mu_0_U[i].flatten(), nu_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten()))  )) for i in range(n_hidden_1)]
        
    zeta_0_U = np.asarray(zeta_0_U) + la_0_U
    
    
    #Initialising variable for main loop
    curr_zeta_L = zeta_0_L
    curr_zeta_U = zeta_0_U
    curr_mu_L = mu_0_L
    curr_mu_U = mu_0_U
    curr_nu_L = [nu_0_L]
    curr_nu_U = [nu_0_U]
    curr_la_L = la_0_L
    curr_la_U = la_0_U
    
    W_Ls = W_0_L.flatten()
    W_Us = W_0_U.flatten()
    #loop over the hidden layers
    for l in range(1,n_hidden_layers+1):
        if l < n_hidden_layers:
            curr_n_hidden = widths[l]
        else:
            curr_n_hidden = 1
            
        LUB = np.asarray(get_alphas_betas(curr_zeta_L, curr_zeta_U))
        LUB = np.asmatrix(LUB).transpose() 
        # Now evaluate eq (*) conditions:
        curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar = get_bar_lower(LUB, curr_mu_L, curr_mu_U, 
                                                           curr_nu_L, curr_nu_U, 
                                                           curr_la_L, curr_la_U)

        curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar = get_bar_upper(LUB, curr_mu_L, curr_mu_U, 
                                                           curr_nu_L, curr_nu_U, 
                                                           curr_la_L, curr_la_U)
        
        curr_z_L = [   min_of_linear_fun( [LUB[i,2]] , [curr_zeta_U[i]] , [curr_zeta_L[i]]     ) + LUB[i,3]
                      for i in range(len(curr_zeta_U))    ]

        #SUpper and lower bounds for weights and biases of current hidden layer
        curr_W_L, curr_W_U, curr_b_L, curr_b_U = (sWs[l][0] - dWs[l]*w_margin,  sWs[l][0] + dWs[l]*w_margin,
                                      sbs[l][0] - dbs[l]*w_margin, sbs[l][0] + dbs[l]*w_margin)
    
        a_L, b_L, c_L = get_abc_lower(curr_W_L, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                               curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)
        
        a_U, b_U, c_U = get_abc_upper(curr_W_U, curr_mu_L_bar, curr_nu_L_bar, curr_la_L_bar,
                               curr_mu_U_bar, curr_nu_U_bar, curr_la_U_bar)
        
        curr_mu_L = np.sum(a_L, axis=0)
        curr_mu_U = np.sum(a_U, axis=0)
        curr_nu_L = []
        curr_nu_U = []
        for k in range(l-1):
            curr_nu_L.append(np.sum(b_L[k], axis=1))
            curr_nu_U.append(np.sum(b_U[k], axis=1))
        
        curr_nu_L.append(b_L[l-1])
        curr_nu_U.append(b_U[l-1])
        
        
        
        
        curr_nu_L.append(np.asarray([curr_z_L for i in range(curr_n_hidden) ]))
        curr_nu_U.append(np.asarray([curr_z_L for i in range(curr_n_hidden) ]))
        
            
        curr_la_L = np.sum(c_L, axis=0) - np.dot(curr_z_L, curr_W_L) + curr_b_L
        curr_la_U = np.sum(c_U, axis=0) - np.dot(curr_z_L, curr_W_U) + curr_b_U
    

            
        curr_zeta_L = []
        curr_zeta_U = []
        
        for i in range(curr_n_hidden):
            ith_mu_L = curr_mu_L[i]
            ith_mu_U = curr_mu_U[i]

            
            ith_W_Ls = np.concatenate( (W_Ls, curr_W_L.T[i]) )
            ith_W_Us = np.concatenate( (W_Us, curr_W_U.T[i]) )
            ith_nu_L = []
            ith_nu_U = []
            for k in range(len(curr_nu_L)):
                ith_nu_L = np.concatenate(  ( ith_nu_L, np.asarray(curr_nu_L[k][i]).flatten()  )    )
                ith_nu_U = np.concatenate(  ( ith_nu_U, np.asarray(curr_nu_U[k][i]).flatten()  )    )
                
               
            curr_zeta_L.append( min_of_linear_fun( np.concatenate( (ith_mu_L, ith_nu_L) ) ,
                                                       np.concatenate( (x_u, ith_W_Us     ) ) ,
                                                       np.concatenate( (x_l, ith_W_Ls     ) )
                                                      )   )  
            
            curr_zeta_U.append( max_of_linear_fun( np.concatenate( (ith_mu_U, ith_nu_U) ) ,
                                                   np.concatenate( (x_u, ith_W_Us     ) ) ,
                                                   np.concatenate( (x_l, ith_W_Ls     ) )
                                                  )   ) 
        curr_zeta_L  = curr_zeta_L + curr_la_L
        curr_zeta_U  = curr_zeta_U + curr_la_U
        
        W_Ls = np.concatenate((W_Ls ,   curr_W_L.T.flatten()  ))
        W_Us = np.concatenate((W_Us ,   curr_W_U.T.flatten()  ))
        
    #Code adaptation for output:    
    #end code adaptation for output
    return [curr_zeta_L, curr_zeta_U]
            
            
"""
A simple conversion from Andrea's outpt to the probablility... plus making it multiple
samples. 
"""
def linear_propogation_VCAS(x, in_reg, out_ind, 
                            w_margin=0.25, search_samps=500, act = 'relu', 
                            reverse=False):
    try:
        loaded_model = np.load(model_path, allow_pickle=True)
        [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    except:
        with open(model_path, 'r') as pickle_file:
            [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = pickle.load(pickle_file)
    sW_0 = np.random.normal(mW_0, dW_0, (search_samps, mW_0.shape[0], mW_0.shape[1]))
    sb_0 = np.random.normal(mb_0, db_0, (search_samps, mb_0.shape[0]))
    sW_1 = np.random.normal(mW_1, dW_1, (search_samps, mW_1.shape[0], mW_1.shape[1]))
    sb_1 = np.random.normal(mb_1, db_1, (search_samps, mb_1.shape[0]))
    valid_weight_intervals = []
    for i in trange(search_samps):
        # Full forward pass in one line :-)
        if act == 'tanh':
            y = (tf.matmul(tf.tanh(tf.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i]).eval()
        elif act == 'relu':
            y = (np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])
        # Logical check if weights sample out_reg constraint
        print (np.matmul(my_relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])
        extra_gate = (reverse and np.argmax(y) != out_ind)
        if(np.argmax(y) == out_ind or extra_gate):
            
            [y_pred_l, y_pred_u] = propogate_lines(x, in_reg, [[sW_0[i]] ,[sW_1[i]] ] , [[sb_0[i]] ,[sb_1[i]] ], w_margin, act )

            y_pred_l = np.squeeze(y_pred_l); y_pred_u = np.squeeze(y_pred_u)
            print y_pred_l
            print y_pred_u
            print y_pred_l <= y
            print y_pred_u >= y
            assert((y_pred_l <= y).all())
            assert((y_pred_u >= y).all()) 
            extra_gate = (reverse and np.argmax(y_pred_l) != out_ind and np.argmax(y_pred_u) != out_ind)
            if((np.argmax(y_pred_l) == out_ind and np.argmax(y_pred_u) == out_ind) or extra_gate):
                # If it does, add the weight to the set of valid weights
                valid_weight_intervals.append([sW_0[i], sb_0[i], sW_1[i], sb_1[i]])
    print "We found %s many valid intervals."%(len(valid_weight_intervals))
    if(len(valid_weight_intervals) == 0):
        return 0.0
    # Now we need to take all of the valid weight intervals we found and merge them
    # so we seperate the valid intervals into their respective variables
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    for v in valid_weight_intervals:
        vW_0.append(v[0])
        vb_0.append(v[1])
        vW_1.append(v[2])
        vb_1.append(v[3])
    # After we merge them, we need to use the erf function to evaluate exactly what the 
    #   lower bound on the probability is!
    pW_0 = compute_interval_probs_weight(np.asarray(vW_0), marg=w_margin, mean=mW_0, std=dW_0)
    pb_0 = compute_interval_probs_bias(np.asarray(vb_0), marg=w_margin, mean=mb_0, std=db_0)
    pW_1 = compute_interval_probs_weight(np.asarray(vW_1), marg=w_margin, mean=mW_1, std=dW_1)
    pb_1 = compute_interval_probs_bias(np.asarray(vb_1), marg=w_margin, mean=mb_1, std=db_1)
    
    # Now that we have all of the probabilities we just need to multiply them out to get
    # the final lower bound on the probability of the condition holding.
    # Work with these probabilities in log space
    p = 0.0
    for i in pW_0.flatten():
        p+=math.log(i)
    for i in pb_0.flatten():
        p+=math.log(i)
    for i in pW_1.flatten():
        p+=math.log(i)
    for i in pb_1.flatten():
        p+=math.log(i)
    #print math.exp(p)
    return math.exp(p)
