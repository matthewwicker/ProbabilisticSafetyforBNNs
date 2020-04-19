import numpy as np
import tensorflow as tf
import ProbablisticReachability
from ProbablisticReachability import linear_propogation_VCAS
ProbablisticReachability.set_model_path("VCAS_MODEL_512.net")

"""
PHI 1 - CONSISTENCY OF DES1500 WARNING
"""
# Hyper Rectangle Test Cases
hyper_rect_1_1 = [[0.10,0.15], [0.01, 0.3], [-0.2,0.15], [-0.3,0.1]]
#hyper_rect_1_1 = [[0.1245,0.1255], [0.199, 0.201], [-0.01,0.01], [-0.06,-0.04]]
hyper_rect_1_2 = [[0.05,0.075], [0.1,0.25], [-0.2, -0.10], [-0.35,-0.20]]
hyper_rect_1_3 = [0.0,0.075], [0.1,0.25], [-0.2,0.10], [-0.2,0.45]

x1 = [[0.125, 0.2, 0.0, -0.05]]
x2 = [[0.075, 0.2, -0.10, -0.05]]
x3 = [[0.075, 0.2, 0.00, -0.05]]
#x3 = [[0.0, 0.1, -0.20, 0.45]]
x_reg_1 = np.transpose(hyper_rect_1_1)
x_reg_2 = np.transpose(hyper_rect_1_2)
x_reg_3 = np.transpose(hyper_rect_1_3)

out_cls = 1
ph1 = 1-linear_propogation_VCAS(x1, x_reg_1, out_cls, w_margin=0.01, search_samps=1500)
no = fuck
print ph1
ph2 = 1-linear_propogation_VCAS(x2, x_reg_2, out_cls, w_margin=2.5, search_samps=1500)
print ph2
ph3 = 1-linear_propogation_VCAS(x3, x_reg_3, out_cls, w_margin=2.5, search_samps=1500)
print "Hyper Rectangle Probabilities:"
print ph1, ph2, ph3
print "============================================="
print " 80% DES1500 Consistency Lowerbound:  "
print "============================================="
print "DES1500: %s"%(1 - (ph1+ph2+ph3))
print "============================================="

"""
PHI 2 - CONSISTENCY OF CLI1500 WARNING
"""
# Hyper Rectangle Test Cases
hyper_rect_2_1 = [[-0.10,-0.15], [-0.1,-0.3], [-0.1,0.15], [-0.3,0.1]]
#[[-0.10,-0.15], [-0.1,-0.3], [-0.1,0.15], [-0.2,0.1]]
hyper_rect_2_2 = [[0.0,-0.05], [-0.1,-0.25], [-0.45, -0.10], [-0.45,-0.20]]
hyper_rect_2_3 = [0.0,-0.075], [-0.1,-0.25], [-0.1,0.20], [-0.2,0.45]

x1 = [[-0.125, -0.2, 0.0, -0.05]]
x2 = [[-0.075, -0.2, -0.10, -0.05]]
x3 = [[-0.075, -0.2, 0.00, -0.05]]
#x3 = [[0.0, 0.1, -0.20, 0.45]]
x_reg_1 = np.transpose(hyper_rect_1_1)
x_reg_2 = np.transpose(hyper_rect_1_2)
x_reg_3 = np.transpose(hyper_rect_1_3)

out_cls = 2
ph1 = 1-linear_propogation_VCAS(x1, x_reg_1, out_cls, w_margin=2.5, search_samps=1500)
print ph1
ph2 = 1-linear_propogation_VCAS(x2, x_reg_2, out_cls, w_margin=2.5, search_samps=1500)
print ph2
ph3 = 1-linear_propogation_VCAS(x3, x_reg_3, out_cls, w_margin=2.5, search_samps=1500)
print "Hyper Rectangle Probabilities:"
print ph1, ph2, ph3
print "============================================="
print " 80% CLI1500 Consistency Lowerbound:  "
print "============================================="
print "CLI1500: %s"%(1 - (ph1+ph2+ph3))
print "============================================="


"""
PHI 3 - NO DANGEROUS DIS1500 WARNINGS
"""
hyper_rect_1_R = [[-0.1,-0.55], [-0.1,-0.5], [-0.2,0.15], [-0.5,0.5]]
x1_R = [[-0.25, -0.25, 0.0, 0.0]]
x_reg_r = np.transpose(hyper_rect_1_R)

out_cls = 1
ph1_r = linear_propogation_VCAS(x1_R, x_reg_r, out_cls, w_margin=2.75, search_samps=1000, reverse=True)
print "============================================="
print " NO DANGEROUS DES1500:  "
print "============================================="
print "NO D-DES1500: %s"%(ph1_r)
print "============================================="
"""
PHI 4 - CONSISTENCY OF DNC WARNING
"""
hyper_rect_2_R = [[0.05,0.55], [0.1,0.5], [-0.2,0.15], [-0.5,0.5]]
x1_R = [[0.25, 0.25, 0.0, 0.0]]
x_reg_r = np.transpose(hyper_rect_2_R)

out_cls = 2
ph2_r = linear_propogation_VCAS(x1_R, x_reg_r, out_cls, w_margin=2.75, search_samps=1000, reverse=True)
print "============================================="
print " NO DANGEROUS CLI1500:  "
print "============================================="
print "NO D-CLI1500: %s"%(ph2_r)
print "============================================="
