
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
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

def model_normal(X, Y, mu_0, sigma_prior, sigma_rand_error):

    for i in range (len(mu_0)):
        coefficient_prior = dist.Normal(mu_0[i], sigma_prior)
        beta_coef = pyro.sample(f"beta_{i}", coefficient_prior)
        linear_combination = X[:, i] * beta_coef
    
    
    # Define a sigma prior for the random error
    sigma = pyro.sample("sigma", dist.HalfNormal(scale=sigma_rand_error))
    
    # For a simple linear model, the expected mean is the linear combination of parameters
    mean = linear_combination
    
    
    with pyro.plate("data", Y.shape[0]):
        
        # Assume our expected mean comes from a normal distribution with the mean which
        # depends on the linear combination, and a standard deviation "sigma"
        outcome_dist = dist.Normal(mean, sigma)
        
        # Condition the expected mean on the observed target y
        observation = pyro.sample("obs", outcome_dist, obs=Y)

def MCMC_Bayesian_update (X, Y, model, mu_0, sigma_prior, sigma_rand_error, 
            n_mcmc = 100, warmup_steps = 20, max_tree_depth=5):

    X_torch = torch.tensor(X.values)
    Y_torch = torch.tensor(Y.values)

    # Clear the parameter storage
    pyro.clear_param_store()

    # Initialize our No U-Turn Sampler
    kernel = NUTS(model, 
                    max_tree_depth = max_tree_depth) # a shallower tree helps the algorithm run faster

    # Employ the sampler in an MCMC sampling 
    mcmc = MCMC(kernel,
                    num_samples=n_mcmc,
                    warmup_steps= warmup_steps)


    # Let's time our execution as well
    start_time = time.time()

    # Run the sampler
    mcmc.run(X_torch, Y_torch, mu_0, 
                sigma_prior, sigma_rand_error)

    end_time = time.time()

    print(f'Inference ran for {round((end_time -  start_time)/60.0, 2)} minutes')

    return mcmc