import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
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
from eig_comp_utils import *


if __name__ == "__main__":

    rng = np.random.RandomState(42)
    show_plots = False

    n_host_and_mirror = 200
    X0 = np.random.randint(0, 2, size=n_host_and_mirror)
    X1 = np.random.normal(size=n_host_and_mirror)
    x_distributions = {0: X0, 1: X1}

    p_assigned_to_host = lambda X_1, X_2, T, eps: sigmoid(1 + X_1 - X_2 + T + eps)
    p_assigned_to_cand2 = lambda X_1, X_2, T, eps: sigmoid(1 - 3 * X_1 + eps)
    X_rct, T_rct = generate_rct(n_host_and_mirror, x_distributions)

    data_host, data_mirror = generate_host_and_mirror(
        X_rct, T_rct, p_assigned_to_cand2
    )  # Jake: Should this be p_assigned_to_host?
    design_data_host = generate_design_matrix(data_host, power_x=1, power_x_t=1)
    design_data_mirror = generate_design_matrix(data_mirror, power_x=1, power_x_t=1)

    n_pre_cand2 = 500
    pre_X_cand2, pre_T_cand2 = generate_rct(n_pre_cand2, x_distributions)
    data_cand2 = generate_host2(
        pre_X_cand2, pre_T_cand2, p_assigned_to_cand2, n_cand2=100
    )
    design_data_cand2 = generate_design_matrix(data_cand2, power_x=1, power_x_t=1)

    outcome_function = (
        lambda X, T, eps: 1
        + 0.5 * X[:, 0]
        + 2 * X[:, 1]
        - 12 * T
        - 6 * X[:, 1] * T
        + eps
    )  # old version 1 / (1 + np.exp(-(1 + 2 * X[0] + 3 * X[1] + 5 * T - 6* X[1]*T eps)))
    design_data_host = add_outcome(design_data_host, outcome_function)
    design_data_mirror = add_outcome(design_data_mirror, outcome_function)
    design_data_cand2 = add_outcome(design_data_cand2, outcome_function)

    # Initialize prior parameters
    beta_0, sigma_0_sq = np.array([-0.5, 4.5, 7.5, -4.5, 1, 12]), 1
    prior_hyperparameters = {"beta_0": beta_0, "sigma_0_sq": sigma_0_sq}
    bayes_reg = BayesianLinearRegression(prior_hyperparameters)

    ### Bayesian update through host dataset
    X_host, Y_host = design_data_host.drop(columns=["Y"]), design_data_host["Y"]
    post_host_parameters = bayes_reg.fit(X_host, Y_host)

    # Generate Y_prior
    sigma_prior = 1  # Standard deviation for Y_prior
    Y_prior = np.dot(X_host, beta_0) + np.random.normal(0, sigma_prior, len(X_host))

    # Generate Y_post_host
    beta_post_host = post_host_parameters[
        "posterior_mean"
    ].flatten()  # Extract posterior mean
    cov_matrix_post_host = post_host_parameters["posterior_cov_matrix"]
    Y_post_host = np.dot(X_host, beta_post_host) + np.random.normal(
        0, 1, len(X_host)
    )  # Assuming standard deviation for Y_post_host is 1

    plot_densities(
        Y_prior,
        Y_post_host,
        design_data_host["Y"],
        names=["Y_prior", "Y_post_host", "True Y"],
        title="Y_post_host vs Y_prior vs True Y",
    )

    ### Bayesian update through candidate datasets
    sigma_cand = 1
    prior_hyperparameters_cand = {"beta_0": beta_post_host, "sigma_0_sq": sigma_cand}

    ## With candidate = mirror dataset
    bayes_reg_mirror = BayesianLinearRegression(prior_hyperparameters_cand)
    X_mirror, Y_mirror = design_data_mirror.drop(columns=["Y"]), design_data_mirror["Y"]
    post_mirror_parameters = bayes_reg_mirror.fit(X_mirror, Y_mirror)

    # Generate Y_post_mirror
    post_mirror_mean = post_mirror_parameters[
        "posterior_mean"
    ].flatten()  # Extract posterior mean
    Y_post_mirror = np.dot(X_mirror, post_mirror_mean) + np.random.normal(
        0, 1, len(X_mirror)
    )  # Assuming standard deviation for Y_post_host is 1

    ## With candidate = cand2 dataset
    bayes_reg_cand2 = BayesianLinearRegression(prior_hyperparameters_cand)
    X_cand2, Y_cand2 = design_data_cand2.drop(columns=["Y"]), design_data_cand2["Y"]
    post_cand2_parameters = bayes_reg_cand2.fit(X_cand2, Y_cand2)

    # Generate Y_post_cand2
    post_cand2_mean = post_cand2_parameters[
        "posterior_mean"
    ].flatten()  # Extract posterior mean
    Y_post_cand2 = np.dot(X_cand2, post_cand2_mean) + np.random.normal(
        0, 1, len(X_cand2)
    )  # Assuming standard deviation for Y_post_host is 1

    plot_densities(
        Y_post_mirror,
        Y_post_cand2,
        design_data_host["Y"],
        names=["Y_post_mirror", "Y_post_cand2", "True Y"],
        title="Y_post_mirror vs Y_post_cand2 vs True Y",
    )

    print("done")


##### JAKE NEW: Added from here, this is the right formula but EIG does not work for log denom
### Need to make some changes for numerical stability

# Number of samples used to estimate outer expectation
n_outer_expectation = 10
m_inner_expectation = int(np.ceil(np.sqrt(n_outer_expectation)))
# Incorporating sqrt constraint into MCMC samples
n_mcmc = n_outer_expectation * (m_inner_expectation + 1)

warmup_steps = 5
max_tree_depth = 7
sigma_rand_error = 1

## Bayesian update using the host dataset
mcmc_host = MCMC_Bayesian_update(
    X=X_host,
    Y=Y_host,
    model=model_normal,
    mu_0=beta_0,
    sigma_prior=sigma_prior,
    sigma_rand_error=sigma_rand_error,
    n_mcmc=n_mcmc,
    warmup_steps=warmup_steps,
    max_tree_depth=max_tree_depth,
)
mcmc_host.summary()

beta_post_host = pd.DataFrame(mcmc_host.get_samples())

# Shuffling to remove any dependence between adjacent samples
beta_post_host = beta_post_host.sample(frac=1)

beta_post_host.head()
# We delete the column with the std
beta_post_host = beta_post_host.iloc[:, :-1]


# EIG computation : observational

eig_obs_closed_form = compute_EIG_obs_closed_form(
    X_mirror, cov_matrix_post_host, sigma_rand_error
)
print(
    "Observational EIG is equal to "
    + str(eig_obs_closed_form)
    + " when computed from closed-form expressions"
)

Y_pred_candidate = predict_with_all_sampled_linear(beta_post_host, X_mirror)
pred_list = samples_in_EIG_form(
    Y_pred_candidate, n_outer_expectation, m_inner_expectation
)
eig_obs_from_samples = compute_EIG_obs_from_samples(pred_list, sigma_rand_error)
print(
    "Observational EIG is equal to "
    + str(eig_obs_from_samples)
    + " when computed from samples"
)

# EIG computation : causal

causal_param_first_index = 3
eig_causal_closed_form = compute_EIG_causal_closed_form(
    X_mirror, cov_matrix_post_host, sigma_rand_error, causal_param_first_index
)
print(
    "Observational EIG is equal to "
    + str(eig_causal_closed_form)
    + " when computed from closed-form expressions"
)

print("done")
