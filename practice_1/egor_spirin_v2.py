# There should be no main() in this file!!!
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1,
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,)
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

# In variant 2 the following functions are required:

import numpy as np
from scipy.stats import binom, poisson


def pa(params, model, only_val=False):
    n = params["amax"] - params["amin"] + 1
    if only_val:
        return 1 / n
    val = np.arange(params["amin"], params["amax"] + 1)
    prob = np.full(fill_value=1 / n, shape=(n,))
    return prob, val


def pb(params, model, only_val=False):
    n = params["bmax"] - params["bmin"] + 1
    if only_val:
        return 1 / n
    val = np.arange(params["bmin"], params["bmax"] + 1)
    prob = np.full(fill_value=1 / n, shape=(n,))
    return prob, val


def pc_ab(a, b, params, model):
    c_max = params["amax"] + params["bmax"]
    prob = np.zeros((c_max + 1, a.shape[0], b.shape[0]))
    val = np.arange(c_max + 1)

    if model == 1 or model == 3:
        # [c max; a size]
        pmf_a = binom.pmf(val.reshape(-1, 1), a, params["p1"])
        # [c max; b size]
        pmf_b = binom.pmf(val.reshape(-1, 1), b, params["p2"])

        for k in range(c_max + 1):
            # P(c = k | a, b) = sum_{i} p(a=i) * p(b=k-i)
            prob[k, :, :] = pmf_a[: k + 1].T @ pmf_b[: k + 1][::-1]
    else:
        # [a size; b size]
        ps_lambda = a.reshape(-1, 1) * params["p1"] + b * params["p2"]
        # [c max; a size; b size]
        prob = poisson.pmf(np.expand_dims(val, (1, 2)), mu=ps_lambda)

    return prob, val


def pc(params, model):
    a = np.arange(params["amin"], params["amax"] + 1)
    b = np.arange(params["bmin"], params["bmax"] + 1)

    prob_c_ab, val = pc_ab(a, b, params, model)
    # p(c) = sum_{a, b} p(c | a, b) * p(a) * p(b)
    # [c max]
    prob = prob_c_ab.sum((1, 2)) * pa(params, model, only_val=True) * pb(params, model, only_val=True)

    return prob, val


def pd_c(c, params, model):
    d_max = 2 * (params["amax"] + params["bmax"])
    val = np.arange(d_max + 1)

    # [d max; c size]
    k = np.arange(d_max + 1).reshape(-1, 1) - c
    # [d max; c size]
    prob = binom.pmf(k, c, params["p3"])

    return prob, val


def pd(params, model):
    prob_c, val_c = pc(params, model)
    prob_d_c, val_d = pd_c(val_c, params, model)

    # p(d) = sum_{c} p(d | c) p(c)
    prob = prob_d_c.dot(prob_c)

    return prob, val_d


def pc_a(a, params, model):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    prob_c_ab, val = pc_ab(a, b, params, model)
    # p(c | a) = sum_{a} p(c | a, b) * p(b)
    # [c max; a size]
    prob = prob_c_ab.sum(2) * pb(params, model, only_val=True)
    return prob, val


def pc_b(b, params, model):
    a = np.arange(params["amin"], params["amax"] + 1)
    prob_c_ab, val = pc_ab(a, b, params, model)
    # p(c | b) = sum_{b} p(c | a, b) * p(a)
    # [c max; b size]
    prob = prob_c_ab.sum(1) * pa(params, model, only_val=True)
    return prob, val


def pb_a(a, params, model):
    prob_b, val_b = pb(params, model)
    prob = np.repeat(prob_b.reshape(-1, 1), a.shape[0], axis=-1)
    return prob, val_b


def pd_b(b, params, model):
    c = np.arange(params["amax"] + params["bmax"] + 1)
    prob_d_c, val_d = pd_c(c, params, model)
    prob_c_b, val_c = pc_b(b, params, model)
    # p(d | b) = sum_{c} p(d | c) * p(c | b)
    prob = prob_d_c.dot(prob_c_b)
    return prob, val_d


def pb_d(d, params, model):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    # [d size; b size]
    prob_d_b = pd_b(b, params, model)[0][d]

    # [b size; d size]
    # p(b | d) = p(d | b) * p(b) / p(d)
    # p(d) = sum_{b} p(d | b) * p(b)
    prob = prob_d_b.T * pb(params, model, only_val=True)
    prob /= prob.sum(axis=0)

    return prob, b


def pb_ad(a, d, params, model):
    b = np.arange(params["bmin"], params["bmax"] + 1)
    c = np.arange(params["amax"] + params["bmax"] + 1)

    # [d size; c size]
    prob_d_c = pd_c(c, params, model)[0][d]
    # [c max; a size; b size]
    prob_c_ab = pc_ab(a, b, params, model)[0]
    # [c max; a size]
    prob_c_a = pc_a(a, params, model)[0]

    # sum_{c} p(d | c) p(c | a,b) * p(b)
    # [d size; a size; b size]
    numerator = prob_d_c.dot(prob_c_ab.transpose((1, 0, 2))) * pb(params, model, only_val=True)
    # sum_{c} p(d | c) p(c | a)
    # [d size; a size]
    denominator = prob_d_c.dot(prob_c_a)

    # [d size; a size; b size]
    prob = numerator / denominator[..., None]
    # [b size; a size; d size]
    prob = prob.transpose((2, 1, 0))

    return prob, b
