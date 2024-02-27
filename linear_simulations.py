import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import time
from sklearn.neighbors import KernelDensity

__name__ = '__main__'

from RCT_experiment import *
from Bayes_linear_regression import *
from plotting_functions import *
from MCMC_Bayesian_update import *
from utils import *


if __name__ == '__main__':

    rng = np.random.RandomState(42)
    show_plots = False

    n_host_and_mirror = 200
    X0 = np.random.randint(0, 2, size=n_host_and_mirror)
    X1 = np.random.normal(size=n_host_and_mirror)
    x_distributions = {0: X0, 1: X1}

    p_assigned_to_host = lambda X_1, X_2, T, eps: sigmoid(1 + X_1 - X_2 + T + eps)
    p_assigned_to_cand2 = lambda X_1, X_2, T, eps: sigmoid(1 - 3*X_1 + eps)
    X_rct, T_rct = generate_rct(n_host_and_mirror, x_distributions)

    data_host, data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2) # Jake: Should this be p_assigned_to_host?
    design_data_host = generate_design_matrix(data_host, power_x=1, power_x_t=1)
    design_data_mirror = generate_design_matrix(data_mirror, power_x=1, power_x_t=1)

    n_pre_cand2 = 500
    pre_X_cand2, pre_T_cand2 = generate_rct(n_pre_cand2, x_distributions)
    data_cand2 = generate_host2(pre_X_cand2, pre_T_cand2, p_assigned_to_cand2, n_cand2=100)
    design_data_cand2 = generate_design_matrix(data_cand2, power_x=1, power_x_t=1)

    outcome_function = lambda X, T, eps: 1 + 0.5 * X[:,0] + 2 * X[:,1] - 12 * T - 6* X[:,1]*T + eps # old version 1 / (1 + np.exp(-(1 + 2 * X[0] + 3 * X[1] + 5 * T - 6* X[1]*T eps)))
    design_data_host = add_outcome(design_data_host, outcome_function)
    design_data_mirror = add_outcome(design_data_mirror, outcome_function)
    design_data_cand2 = add_outcome(design_data_cand2, outcome_function)

    # Initialize prior parameters
    beta_0, sigma_0_sq = np.array([-0.5, 4.5, 7.5, -4.5, 1, 12]), 1
    prior_hyperparameters = {'beta_0': beta_0, 'sigma_0_sq': sigma_0_sq}
    bayes_reg = BayesianLinearRegression(prior_hyperparameters)

    ### Bayesian update through host dataset
    X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
    post_host_parameters = bayes_reg.fit(X_host, Y_host)

    # Generate Y_prior
    sigma_prior = 1  # Standard deviation for Y_prior
    Y_prior = np.dot(X_host, beta_0) + np.random.normal(0, sigma_prior, len(X_host))

    # Generate Y_post_host
    beta_post_host = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
    Y_post_host = np.dot(X_host, beta_post_host) + np.random.normal(0, 1, len(X_host))  # Assuming standard deviation for Y_post_host is 1

    plot_densities(Y_prior, Y_post_host, design_data_host['Y'], 
                   names = ['Y_prior', 'Y_post_host', 'True Y'], 
                   title = 'Y_post_host vs Y_prior vs True Y')

    ### Bayesian update through candidate datasets
    sigma_cand = 1
    prior_hyperparameters_cand = {'beta_0': beta_post_host, 'sigma_0_sq': sigma_cand}
    bayes_reg_cand = BayesianLinearRegression(prior_hyperparameters_cand)

    ## With candidate = mirror dataset
    X_mirror, Y_mirror = design_data_mirror.drop(columns=['Y']), design_data_mirror['Y']
    post_mirror_parameters = bayes_reg.fit(X_mirror, Y_mirror)

    # Generate Y_post_mirror
    post_mirror_mean = post_mirror_parameters['posterior_mean'].flatten()  # Extract posterior mean
    Y_post_mirror = np.dot(X_mirror, post_mirror_mean) + np.random.normal(0, 1, len(X_mirror))  # Assuming standard deviation for Y_post_host is 1

    ## With candidate = cand2 dataset
    X_cand2, Y_cand2 = design_data_cand2.drop(columns=['Y']), design_data_cand2['Y']
    post_cand2_parameters = bayes_reg.fit(X_cand2, Y_cand2)
    
    # Generate Y_post_cand2
    post_cand2_mean = post_cand2_parameters['posterior_mean'].flatten()  # Extract posterior mean
    Y_post_cand2 = np.dot(X_cand2, post_cand2_mean) + np.random.normal(0, 1, len(X_cand2))  # Assuming standard deviation for Y_post_host is 1

    plot_densities(Y_post_mirror, Y_post_cand2, design_data_host['Y'],
                   names = ['Y_post_mirror', 'Y_post_cand2', 'True Y'], 
                   title = 'Y_post_mirror vs Y_post_cand2 vs True Y')

    print('done')


##### JAKE NEW: Added from here, this is the right formula but EIG does not work for log denom
    ### Need to make some changes for numerical stability 

#Number of samples used to estimate outer expectation
n_samples_for_expectation = 10
m_samples_for_expectation = int(np.ceil(np.sqrt(n_samples_for_expectation)))
# Incorporating sqrt constraint into MCMC samples
n_mcmc = (n_samples_for_expectation * (m_samples_for_expectation+1)) 

warmup_steps = 5
max_tree_depth = 7
sigma_rand_error = 1

## Bayesian update using the host dataset
mcmc_host = MCMC_Bayesian_update(X =X_host, Y= Y_host, model =model_normal,
            mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
            n_mcmc = n_mcmc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)
mcmc_host.summary()

beta_post_host = pd.DataFrame(mcmc_host.get_samples())

#Shuffling to remove any dependence between adjacent samples
beta_post_host = beta_post_host.sample(frac = 1)

beta_post_host.head()
# We delete the column with the std
beta_post_host = beta_post_host.iloc[:, :-1] 

# I will now do a version of computing the EIG at the mirror dataset, first we form the mirror predictions for each sampled beta:
Y_pred_candidate = predict_with_all_sampled_betas(beta_post_host, X_mirror)

#Now for the first n_samples_for_expectation terms we sample a paired imagined Y val
Y_sampled_candidate = Y_pred_candidate[:,:n_samples_for_expectation] +  np.random.normal(0, 1, size = Y_pred_candidate[:,:n_samples_for_expectation].shape)

# Now we want to compute the outer expectation, this vector will contain samples of log ratio:
log_ratio_samples = []

# Giving a standard covariance matrix:
covariance = np.diag(np.ones_like(Y_sampled_candidate))

#Now we are averaging over our a paired y,beta:
for i,(y,y_pred_beta) in enumerate(zip(Y_sampled_candidate.T,(Y_pred_candidate[:,:n_samples_for_expectation].T))):
    print(y.shape)
    print(y_pred_beta.shape)
    covariance = np.diag(np.ones_like(y))
    # First computing the log likelihood of the numerator:

    log_density_numerator = multivariate_normal_log_likelihood(y, y_pred_beta, covariance)

    # Now computing the log likelihood of the denominator, first by getting the samples:
    #Addition ensures no samples are used twice and that we use m_samples_for_expectation for each step

    y_pred_beta_samples = Y_pred_candidate[:, n_samples_for_expectation + i*m_samples_for_expectation:n_samples_for_expectation + (i+1)*m_samples_for_expectation ]
    log_density_denominator = predictive_normal_log_likelihood(y, y_pred_beta_samples, covariance)

    log_ratio_samples.append(log_density_numerator-log_density_denominator)

EIG = sum(log_ratio_samples)/len(log_ratio_samples)


##### JAKE: read from here

n_mcmc = 20
n_noise_over_y = 15
warmup_steps = 5
max_tree_depth = 7
sigma_rand_error = 1

## Bayesian update using the host dataset
mcmc_host = MCMC_Bayesian_update(X =X_host, Y= Y_host, model =model_normal,
                mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
                n_mcmc = n_mcmc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

mcmc_host.summary()
beta_post_host = pd.DataFrame(mcmc_host.get_samples())
### Posterior parameter samples
beta_post_host.head()
# we delete the column with the std
beta_post_host = beta_post_host.iloc[:, :-1] 
    
# Our predictions are the linear combination X*beta = Y using all the beta samples
Y_pred_all_sampled_betas_using_Xhost = predict_with_all_sampled_betas(beta_post_host, X_host)
# Y_pred_all_sampled_betas_using_X_host is of dimension len(X_host) x n_mcmc

Y_pred_all_sampled_betas_using_X_mirror = predict_with_all_sampled_betas(beta_post_host, X_mirror)
# Y_pred_all_sampled_betas_using_X_mirror is of dimension len(X_mirror) x n_mcmc


###### JAKE help needed from here. 
# write down steps with pseudo code please (as detailed as possible)
# anytime there is a new object:
# write its dimensions as a function of n_host, n_mcmc, n_noise, n_cand 
# where n_noise is noise over y

# EIG = expectation [ log(density(numerator)) - log(density(denominator)) ]
# EIG = expectation [log ( P (z | θ, ξ) ) - log ( P (z | ξ) ) ]

## step 1 : A = log(density(numerator))

## step 2 : log(density(denominator))

## step 3 : taking the expectation of [log(density(numerator))] 

## step 4 : taking the expectation of [log(density(denominator))] 



# Notes from earlier, with the formula from the Wikipedia page
# for each beta sample:
# compute Y_pred_one_beta_post_host a vector of size n 
    # in formula replace mu by this Y_pred_one_beta
    # mu is ONE y_predicted 
    # sample noise on y to generate an "unfixed" y
# later average over the number of mcmc_samples 
# ie in some matrix the mean over axis of dim n_mcmc


######## Previous code for NUMERATOR
list_ys = []
for i in range(n_mcmc): # noise over betas
    eps = np.random.normal(size=n_noise_over_y)
    list_log_densities = []
    for j in range(n_noise_over_y): # noise over y
        noise_over_y = eps[j]
        Y_over_noise_one_beta = X_mirror.dot(beta_post_host.iloc[i,:].values) + np.repeat(eps[j], len(X_mirror))
        list_log_densities.append(np.array(Y_over_noise_one_beta).reshape(-1,1))
    list_log_densities = np.hstack(list_log_densities)
    kde_post_cand = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Y_over_noise_one_beta) 
    # following this there was a dimension problem
    log_density_numerator_one_beta = kde_post_cand.score_samples(Y_over_noise_one_beta) 
    list_log_densities.append(log_density_numerator_one_beta) #size n x n_noise_over_y
# list_log_densities has beta elements, each are vectors of size n x n_noise_over_y
list_log_densities = np.stack(list_log_densities, axis=0) 
expected_log_density_numerator = np.mean(list_log_densities, axis=[0,1]) 



# Computing log density of NUMERATOR 

# n is the sample size in candidate dataset
# get L_b values of betas (vectors of size d) from prior (or post host)

    # sample n_noise noise values
    # for each beta : 
        # evaluate Y = X dot beta + noise n 
        # resulting in a matrix M of n * L_n Y values
        # each column is noise over Y, each line is a x sample
        # use the n vectors of size n_noise (of matrix M) to compute a log_kde
        # note: evaluation of log_kde is on matrix M: 
        # log_density_numerator = kde_post_host.score_samples(M)
        # that log_kde vector will be of size n * n_noise
# repeat for the L_b values of beta: resulting in L_b vectors of log_kde of size n 
# take the average on dimensions L_b and n_noise and get a vector of size n 
# this is the expectation of the numerator

# take the kde_post_host and compute the expectation of the denominator using all the Y_values in M 
# using the score samples function



### IGNORE
# Computing log density of DENOMINATOR i.e.before taking the expectation
# P (Y | Xcand, D_host)

# below we need to replace Y_post_host by Y_pred i.e. sample betas and evaluate betas* X_candidate + noise
# we are marginalizing over betas and noise, so we calculate Ys using sampled betas and not a fixed beta

# JAKE so far I do marginalize over betas and X but no procedure wrt noise, is that ok?
# Y_pred_all_sampled_betas_post_host is of size n x n_mcmc
# kde_post_host = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Y_pred_all_sampled_betas_post_host) 



# step 1 get samples from (Y, Beta) / X : do we evaluate each Y from the sampled beta?
# step 2 for each sampled beta we sample the Ys from the distribution of Y knowing beta and X
# (sampling on the independent noise of Y only)
# step 3 for each sampled beta, we get a bunch of Ys: do kde on each bunch of Ys 
# --> this gives an estimation of the log density conditional on beta and X
