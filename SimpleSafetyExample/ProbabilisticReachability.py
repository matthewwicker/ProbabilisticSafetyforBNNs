
import numpy as np
from tqdm import trange
import edward as ed
import tensorflow as tf
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
width = 0
def set_width(w):
    global width
    width = w
    
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

def relu(arr):
    return arr * (arr > 0)
def interval_bound_propagation(x, in_reg, out_reg,
                      w_margin=0.25, search_samps=1000,act = 'tanh'):
    x = np.asarray(x); x = x.astype('float64')
    x_l, x_u = in_reg[0], in_reg[1]
    y_l, y_u = out_reg[0], out_reg[1]
    loaded_model = np.load('VIMODEL_' + str(width) + '_' + act +  '.net.npz', allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    # First, sample and hope some weights satisfy the out_reg constraint
    sW_0 = np.random.normal(mW_0, dW_0, (search_samps, mW_0.shape[0], mW_0.shape[1]))
    sb_0 = np.random.normal(mb_0, db_0, (search_samps, mb_0.shape[0]))
    sW_1 = np.random.normal(mW_1, dW_1, (search_samps, mW_1.shape[0], mW_1.shape[1]))
    sb_1 = np.random.normal(mb_1, db_1, (search_samps, mb_1.shape[0]))
    #pickle.dump([sW_0,sb_0,sW_1,sb_1],open('sampled_w8s.p','w+'))
    valid_weight_intervals = []
    for i in trange(search_samps):
        # Full forward pass in one line :-)
        # WE NEED TO CHANGE THIS FROM TENSORFLOW TO NUMPY...
        if act == 'tanh':
            y = (np.matmul(np.tanh(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])
        elif act == 'relu':
            y = (np.matmul(relu(np.matmul(x, sW_0[i]) + sb_0[i]), sW_1[i]) + sb_1[i])
        # Logical check if weights sample out_reg constraint
        if(y >= y_l and y <= y_u):
            # If so, do interval propagation
            h_l, h_u = propagate_interval(sW_0[i], dW_0, sb_0[i], db_0, x_l, x_u, w_margin)
            #pickle.dump([h_l,h_u],open('pre_act.p','w+'))
            if act == 'tanh':
                h_l, h_u = np.tanh(h_l), np.tanh(h_u)
            elif act == 'relu':
                h_l, h_u = my_relu(h_l), my_relu(h_u)
                #pickle.dump([h_l,h_u],open('post_act.p','w+'))
            y_pred_l, y_pred_u = propagate_interval(sW_1[i], dW_1, sb_1[i], db_1, h_l, h_u, w_margin)
            #pickle.dump([y_pred_l,y_pred_u],open('final_values.p','w+'))
            # Check if interval propagation still respects out_reg constraint
            #print 
            #print y_pred_l
            #print y_pred_u
            #print y_pred_u[0] - y_pred_l[0]
            if(y_pred_l >= y_l and y_pred_u <= y_u):
                # If it does, add the weight to the set of valid weights
                valid_weight_intervals.append([sW_0[i], sb_0[i], sW_1[i], sb_1[i]])
        else:
            # Note that this conditional gives us the ijcai notion for free...
            # could be something useful to note!
            continue
    print "We found %s many valid intervals."%(len(valid_weight_intervals))
    if(len(valid_weight_intervals) == 0):
        return 0.0
    # Now we need to take all of the valid weight intervals we found and merge them
    # so we seperate the valid intervals into their respective variables
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    for v in valid_weight_intervals:
        #np.asarray(v[0]) i removed this... should i not have? -MW
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
    
    
    

# !! THIS IS ONLY FOR RELU !! NO OTHER ACTIVATION SUPPORTED
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
    #nu_l = np.squeeze(nu_l); nu_u = np.squeeze(nu_u); 
    #lam_l = np.squeeze(lam_l); lam_u = np.squeeze(lam_u); 
    #linear_bound_coef= np.squeeze(linear_bound_coef)
    mu_bar, nu_bar, lam_bar = [], [], []
    #coef of the form - alpha_U, beta_U, alpha_L, beta_L
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,2] >= 0):
            mu_bar.append(linear_bound_coef[i,2] * mu_l[i])
            nu_bar.append(linear_bound_coef[i,2] * nu_l[i])
            lam_bar.append(linear_bound_coef[i,2] * lam_l[i] + linear_bound_coef[i,3])
        else:
            mu_bar.append(linear_bound_coef[i,2] * mu_u[i])
            nu_bar.append(linear_bound_coef[i,2] * nu_u[i])
            lam_bar.append(linear_bound_coef[i,2] * lam_u[i] + linear_bound_coef[i,3])
    return np.asarray(mu_bar), np.asarray(nu_bar), np.asarray(lam_bar)

def get_bar_upper(linear_bound_coef, mu_l, mu_u,
                  nu_l, nu_u, lam_l, lam_u):
    mu_l = np.squeeze(mu_l); mu_u = np.squeeze(mu_u);  
    #lam_l = np.squeeze(lam_l); lam_u = np.squeeze(lam_u); 
    mu_bar, nu_bar, lam_bar = [], [], []
    #coef of the form - alpha_U, beta_U, alpha_L, beta_L
    for i in range(len(linear_bound_coef)):
        if(linear_bound_coef[i,0] >= 0):
            mu_bar.append(linear_bound_coef[i,0] * mu_u[i])
            nu_bar.append(linear_bound_coef[i,0] * nu_u[i])
            lam_bar.append(linear_bound_coef[i,0] * lam_u[i] + linear_bound_coef[i,1])
        else:
            mu_bar.append(linear_bound_coef[i,0] * mu_l[i])
            nu_bar.append(linear_bound_coef[i,0] * nu_l[i])
            lam_bar.append(linear_bound_coef[i,0] * lam_l[i] + linear_bound_coef[i,1])
    return np.asarray(mu_bar), np.asarray(nu_bar), np.asarray(lam_bar)

def get_abc_lower(w, mu_l_bar, nu_l_bar, la_l_bar,
               mu_u_bar, nu_u_bar, la_u_bar):
    a, b, c = [], [], []
    for i in range(len(w)):
        for j in range(len(w[i])):
            if(w[i][j] >= 0):
                a.append(w[i][j] * mu_l_bar[i])
                b.append(w[i][j] * nu_l_bar[i])
                c.append(w[i][j] * la_l_bar[i])
            else:
                a.append(w[i][j] * mu_u_bar[i])
                b.append(w[i][j] * nu_u_bar[i])
                c.append(w[i][j] * la_u_bar[i])
    return np.asarray(a), np.asarray(b), np.asarray(c)


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


    
"""
Linear Propogation Function
"""

from my_utils import my_relu
import pickle
def propogate_lines(x, in_reg, out_reg, 
                    w_margin=0.25, search_samps=100, act = 'tanh'):
    x = np.asarray(x); x = x.astype('float64')
    x_l, x_u = in_reg[0], in_reg[1]
    y_l, y_u = out_reg[0], out_reg[1]
    loaded_model = np.load('VIMODEL_' + str(width) + '_' + act +  '.net.npz', allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    # First, sample and hope some weights satisfy the out_reg constraint
    if 1:
        sW_0 = np.random.normal(mW_0, dW_0, (search_samps, mW_0.shape[0], mW_0.shape[1]))
        sb_0 = np.random.normal(mb_0, db_0, (search_samps, mb_0.shape[0]))
        sW_1 = np.random.normal(mW_1, dW_1, (search_samps, mW_1.shape[0], mW_1.shape[1]))
        sb_1 = np.random.normal(mb_1, db_1, (search_samps, mb_1.shape[0]))
    else:
        #using pre-saved w8s for debugging:
        list_of_stuff = pickle.load(open('sampled_w8s.p'))
        [sW_0,sb_0,sW_1,sb_1] =  list_of_stuff
        
        
    #Actual code from now on    
        
    #mW_0, mb_0, mW_1, mb_1 = sW_0[0], sb_0[0], sW_1[0], sb_1[0]
    # Step 1 (According to the algorithmic notes) 
    # NB: Numpy get column i - [:,i]
    
    W_0_L, W_0_U, b_0_L, b_0_U = sW_0[0] - dW_0[0]*w_margin,  sW_0[0] + dW_0[0]*w_margin, sb_0[0]-db_0[0]*w_margin, sb_0[0]+db_0[0]*w_margin
    
    W_0_L = W_0_L.T
    W_0_U = W_0_U.T
    
    mu_0_L = W_0_L; mu_0_U = W_0_U
    
    n_hidden_1 = len(sW_0[0][0]) 
    
    nu_0_L = np.asarray([x_l for i in range(n_hidden_1) ])
    nu_0_U = np.asarray([x_l for i in range(n_hidden_1) ])
    la_0_L = - np.dot(x_l, W_0_L.T) + b_0_L
    la_0_U = - np.dot(x_l, W_0_U.T) + b_0_U
    
    
    # getting bounds on pre-activation fucntion
    zeta_0_L = [ (min_of_linear_fun(np.concatenate((mu_0_L[i].flatten(), nu_0_L[i].flatten())), 
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten() )),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_U[i].flatten() ))  )) for i in range(n_hidden_1)] 
   
    zeta_0_L = np.asarray(zeta_0_L) + la_0_L
     
    zeta_0_U = [ (max_of_linear_fun(np.concatenate((mu_0_U[i].flatten(), nu_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_u).flatten(), W_0_U[i].flatten())),
                                     np.concatenate((np.asarray(x_l).flatten(), W_0_L[i].flatten()))  )) for i in range(n_hidden_1)]
        
    zeta_0_U = np.asarray(zeta_0_U) + la_0_U
    
    # These are our linear upper and lower bounds for the activation function
    LUB = np.asarray(get_alphas_betas(zeta_0_L, zeta_0_U))
    #LUB = np.reshape(LUB, (-1, 4))
    LUB = np.asmatrix(LUB).transpose() 
    # Now evaluate eq (*) conditions:
    mu_0_L_bar, nu_0_L_bar, la_0_L_bar = get_bar_lower(LUB, mu_0_L, mu_0_U, 
                                                       nu_0_L, nu_0_U, 
                                                      la_0_L, la_0_U)

    mu_0_U_bar, nu_0_U_bar, la_0_U_bar = get_bar_upper(LUB, mu_0_L, mu_0_U, 
                                                       nu_0_L, nu_0_U,
                                                      la_0_L, la_0_U)
    
    z_1_L = my_relu(zeta_0_L)
    z_1_U = my_relu(zeta_0_U)
    #z_1_L = [   min_of_linear_fun( [LUB[i,2]] , [zeta_0_U[i]] , [zeta_0_L[i]]     ) + LUB[i,3]
    #              for i in range(n_hidden_1)    ]
    #z_1_U = [   max_of_linear_fun( np.asarray([LUB[i,0]]) , [zeta_0_U[i]] , [zeta_0_L[i]]     ) + LUB[i,1]
    #              for i in range(n_hidden_1)    ]
    
    #pickle.dump([z_1_L,z_1_U],open('post_act_linear.p','w+'))
    #Second layer
    W_1_L, W_1_U, b_1_L, b_1_U = sW_1[0] - dW_1[0]*w_margin,  sW_1[0] + dW_1[0]*w_margin, sb_1[0] - db_1[0]*w_margin, sb_1[0] + db_1[0]*w_margin
    
    a_L, b_L, c_L = get_abc_lower(W_1_L, mu_0_L_bar, nu_0_L_bar, la_0_L_bar,
                           mu_0_U_bar, nu_0_U_bar, la_0_U_bar)

    a_U, b_U, c_U = get_abc_upper(W_1_U, mu_0_L_bar, nu_0_L_bar, la_0_L_bar,
                           mu_0_U_bar, nu_0_U_bar, la_0_U_bar)
    
    mu_1_l = np.sum(a_L, axis=0); mu_1_u = np.sum(a_U, axis=0)
    nu_12_l = z_1_L; nu_12_u = z_1_L
    nu_02_l = b_L; nu_02_u = b_U
    la_1_l = np.sum(c_L, axis=0) - np.dot(z_1_L, W_1_L) + b_1_L
    la_1_u = np.sum(c_U, axis=0) - np.dot(z_1_L, W_1_U) + b_1_U
    
    mu_1_l = np.asarray([mu_1_l])
    out_l = min_of_linear_fun(np.concatenate((mu_1_l.flatten(), nu_02_l.flatten(), nu_12_l)), 
                                np.concatenate((np.asarray(x_u).flatten(), W_0_U.flatten(), W_1_U.flatten())),
                                np.concatenate((np.asarray(x_l).flatten(), W_0_L.flatten(), W_1_L.flatten()))) + la_1_l


    mu_1_u = np.asarray([mu_1_u])
    out_u = max_of_linear_fun(np.concatenate((mu_1_u.flatten(), nu_02_u.flatten(), nu_12_u)), 
                                np.concatenate((np.asarray(x_u).flatten(), W_0_U.flatten(), W_1_U.flatten())),
                                np.concatenate((np.asarray(x_l).flatten(),W_0_L.flatten(), W_1_L.flatten()))) + la_1_u

    #print out_l[0], out_u[0]
    #print 'range: ' + str(out_u[0] - out_l[0])
    return [sW_0,sb_0,sW_1,sb_1],[out_l, out_u]


"""
A simple conversion from Andrea's outpt to the probablility... plus making it multiple
samples. 
"""
def linear_propogation(x, in_reg, out_reg, 
                    w_margin=0.25, search_samps=100, act = 'tanh'):
    loaded_model = np.load('VIMODEL_' + str(width) + '_' + act +  '.net.npz', allow_pickle=True)
    [mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1] = loaded_model['arr_0']
    y_l, y_u = out_reg[0], out_reg[1]
    valid_weight_intervals = []
    for i in trange(search_samps):
        [sW_0,sb_0,sW_1,sb_1],[out_l, out_u] = propogate_lines(x, in_reg, out_reg, w_margin, search_samps, act)
        if(out_l >= y_l and out_u <= y_u):
            #append to valid weights
            valid_weight_intervals.append([sW_0[i], sb_0[i], sW_1[i], sb_1[i]])
            
    print "We found %s many valid intervals."%(len(valid_weight_intervals))
    if(len(valid_weight_intervals) == 0):
        return 0.0
    # Now we need to take all of the valid weight intervals we found and merge them
    # so we seperate the valid intervals into their respective variables
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    for v in valid_weight_intervals:
        #np.asarray(v[0]) i removed this... should i not have? -MW
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