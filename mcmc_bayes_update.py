import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch

torch.set_default_tensor_type(torch.FloatTensor)  # set the default to float32
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import time
from sklearn.neighbors import KernelDensity

__name__ = "__main__"

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from functools import partial


def model_normal(X, Y, mu_0, sigma_prior, sigma_rand_error_fixed, sigma_rand_error):

    linear_combination = 0
    for i in range(len(mu_0)):
        coefficient_prior = dist.Normal(mu_0[i], sigma_prior)
        beta_coef = pyro.sample(f"beta_{i}", coefficient_prior)
        linear_combination += X[:, i] * beta_coef

    # Define a sigma prior for the random error
    if sigma_rand_error_fixed:
        sigma = sigma_rand_error

    else:
        sigma = pyro.sample("sigma", dist.HalfNormal(scale=sigma_rand_error))

    # For a simple linear model, the expected mean is the linear combination of parameters
    mean = linear_combination

    with pyro.plate("data", Y.shape[0]):

        # Assume our expected mean comes from a normal distribution with the mean which
        # depends on the linear combination, and a standard deviation "sigma"
        outcome_dist = dist.Normal(mean, sigma)

        # Condition the expected mean on the observed target y
        observation = pyro.sample("obs", outcome_dist, obs=Y)


def MCMC_Bayesian_update(
    X_torch,
    Y_torch,
    model,
    mu_0,
    sigma_prior,
    sigma_rand_error=1,
    sigma_rand_error_fixed=True,
    n_mcmc=100,
    warmup_steps=20,
    max_tree_depth=5,
):

    # Clear the parameter storage
    pyro.clear_param_store()

    # Initialize our No U-Turn Sampler
    kernel = NUTS(
        model, max_tree_depth=max_tree_depth
    )  # a shallower tree helps the algorithm run faster

    # Employ the sampler in an MCMC sampling
    mcmc = MCMC(kernel, num_samples=n_mcmc, warmup_steps=warmup_steps)

    # Let's time our execution as well
    start_time = time.time()

    # Run the sampler
    mcmc.run(
        X_torch, Y_torch, mu_0, sigma_prior, sigma_rand_error, sigma_rand_error_fixed
    )

    end_time = time.time()

    print(f"Inference ran for {round((end_time -  start_time)/60.0, 2)} minutes")

    return mcmc


def return_causal_samp_func_linear(
    X,
    Y,
    causal_param_first_index,
    mu_0,
    sigma_prior,
    sigma_rand_error,
    sigma_rand_error_fixed,
    warmup_steps,
    max_tree_depth,
):
    mu_0 = mu_0[:causal_param_first_index]

    def causal_samp_func_linear(causal_param, n_non_causal_expectation):
        Y_new = Y - (X[:, causal_param_first_index:] @ causal_param)
        X_new = X[:, :causal_param_first_index]

        mcmc_alg = MCMC_Bayesian_update(
            X_torch=X_new,
            Y_torch=Y_new,
            model=model_normal,
            mu_0=mu_0,
            sigma_prior=sigma_prior,
            sigma_rand_error=sigma_rand_error,
            sigma_rand_error_fixed=sigma_rand_error_fixed,
            n_mcmc=n_non_causal_expectation,
            warmup_steps=warmup_steps,
            max_tree_depth=max_tree_depth,
        )

        beta_nc_samples = pd.DataFrame(mcmc_alg.get_samples())
        return beta_nc_samples

    return causal_samp_func_linear
