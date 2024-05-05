import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import sys
import os, random

notebook_dir = os.getcwd()
parent_dir = os.path.dirname(notebook_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *
from research_exp_utils import *

print('import done')


# rng = np.random.RandomState(42)

# varying_sample_sizes = [100, 120, 140, 160, 180, 200]
# fixed_n_complementary = 100
# n_host = 100 
#  # set to None if both candidates have the same sample size


# n_rct_before_split = 10**5
# std_true_y = 1
# sigma_prior = 1
# sigma_rand_error = 1
# include_intercept = 1  # 0 if no intercept
# power_x, power_x_t = 1, 1

# np.random.seed(42)
# random.seed(42)

# X0 = np.random.beta(12, 3, size=n_rct_before_split)
# X1 = np.random.normal(loc=4, scale=1, size=n_rct_before_split)
# X2 = np.random.beta(1, 7, size=n_rct_before_split)
# x_distributions = {"X_0": X0, "X_1": X1, "X_2": X2}
# d = (
#     include_intercept
#     + len(x_distributions) * (power_x)
#     + 1
#     + len(x_distributions) * (power_x_t)
# )

# p_assigned_to_host = lambda X, T, eps: sigmoid(
#     1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
# )
# p_assigned_to_cand2 = lambda X, T, eps: sigmoid(
#     1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
# )


# causal_param_first_index = power_x*len(x_distributions) + include_intercept 

# outcome_function = (
#     # y = 1 + 1*X_0 - 1*X_1 + 1*X_2 + 4*T + 2*X_0*T + 2*X_1*T + 0*X_2*T + eps
#     lambda X, T, eps: include_intercept  # intercept, non-causal => 0 no intercept
#     + 1 * X["X_0"]  # non-causal
#     - 1 * X["X_1"]  # non-causal
#     + 1 * X["X_2"]  # non-causal
#     + 5 * T  # causal
#     + 2 * X["X_0"] * T  # causal
#     + 2 * X["X_1"] * T  # causal
#     - 4 * X["X_2"] * T  # causal
#     + eps
# )

# CATE_function = lambda X: outcome_function(X,np.ones(len(X)), 0 )-outcome_function(X,np.zeros(len(X)),0)

# # Prior parameters for Bayesian update on host
# if include_intercept:
#     prior_mean = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# else:
#     prior_mean = np.array([0, 0, 0, 0, 0, 0, 0])
# assert len(prior_mean) == d, "Shape error"

# beta_0, sigma_0_sq, inv_cov_0 = (
#     prior_mean,
#     sigma_rand_error**2,
#     1 / sigma_prior * np.eye(len(prior_mean)),
# )
# prior_hyperparameters = {
#     "beta_0": beta_0,
#     "sigma_0_sq": sigma_0_sq,
#     "inv_cov_0": inv_cov_0,
# }

# n_seeds = 50
# data_parameters = {
#     "fixed_n_complementary": fixed_n_complementary,
#     "varying_sample_sizes": varying_sample_sizes,
#     "n_rct_before_split": n_rct_before_split,
#     "x_distributions": x_distributions,
#     "p_assigned_to_cand2": p_assigned_to_cand2,
#     "p_assigned_to_host": p_assigned_to_host,
#     "n_host": n_host,
#     "power_x": power_x,
#     "power_x_t": power_x_t,
#     "outcome_function": outcome_function,
#     "std_true_y": std_true_y,
#     "causal_param_first_index": causal_param_first_index,
# }

# EIG_obs_closed_form_across_seeds, EIG_caus_closed_form_across_seeds = [], []
# store_non_exact_data = {}

# for i in range(n_seeds):
#     nonexact_data = generate_data_varying_sample_size(
#         data_parameters, include_intercept=bool(include_intercept), seed=i)
#     EIGs = linear_eig_closed_form_varying_sample_size(  # CHECK what this does
#         nonexact_data,
#         data_parameters,
#         prior_hyperparameters,
#         verbose=False,
#     )
#     EIG_obs_closed_form_across_seeds.append(
#         [cand_values for cand_values in EIGs[0].values()]
#     )
#     EIG_caus_closed_form_across_seeds.append(
#         [cand_values for cand_values in EIGs[1].values()]
#     )
#     store_non_exact_data[i] = nonexact_data


# EIG_obs_closed_form_across_seeds = np.vstack(EIG_obs_closed_form_across_seeds)  
# EIG_caus_closed_form_across_seeds = np.vstack(EIG_caus_closed_form_across_seeds)


# def turn_into_diff(arr):
#     n, d = np.shape(arr)[0], np.shape(arr)[1]
#     result = np.zeros((n//2, d))
#     for i in range (n//2):
#         result[i,:]=arr[2*i,:]-arr[(2*i) +1,:]
#     return result

# proportions = np.array(varying_sample_sizes)/fixed_n_complementary
# proportions

# dict_diff_EIG_closed_form_across_seeds = {'predictive closed form': turn_into_diff(EIG_obs_closed_form_across_seeds),\
#                                          'causal closed form':turn_into_diff(EIG_caus_closed_form_across_seeds)}


# color_dict = {'predictive closed form': 'green', 'causal closed form': 'red'}

# plot_dict(
#     x = proportions,
#     dict = dict_diff_EIG_closed_form_across_seeds,
#     axis_names=[r'$\frac{n_{twin}}{n_{complementary}}$', 'EIG(comp)-EIG(twin)', "Mean MSE difference"],
#     dict_additional_plots=None,
#     title= None,
#     color_dict=color_dict,
#     save=None
# )