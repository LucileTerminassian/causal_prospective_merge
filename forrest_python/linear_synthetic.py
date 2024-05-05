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


rng = np.random.RandomState(42)

varying_sample_sizes = [100, 120, 140, 160, 180, 200]
fixed_n_complementary = 100
n_host = 100 
 # set to None if both candidates have the same sample size


n_rct_before_split = 10**5
std_true_y = 1
sigma_prior = 1
sigma_rand_error = 1
include_intercept = 1  # 0 if no intercept
power_x, power_x_t = 1, 1

np.random.seed(42)
random.seed(42)

X0 = np.random.beta(12, 3, size=n_rct_before_split)
X1 = np.random.normal(loc=4, scale=1, size=n_rct_before_split)
X2 = np.random.beta(1, 7, size=n_rct_before_split)
x_distributions = {"X_0": X0, "X_1": X1, "X_2": X2}
d = (
    include_intercept
    + len(x_distributions) * (power_x)
    + 1
    + len(x_distributions) * (power_x_t)
)

p_assigned_to_host = lambda X, T, eps: sigmoid(
    1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
)
p_assigned_to_cand2 = lambda X, T, eps: sigmoid(
    1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
)


causal_param_first_index = power_x*len(x_distributions) + include_intercept 

outcome_function = (
    # y = 1 + 1*X_0 - 1*X_1 + 1*X_2 + 4*T + 2*X_0*T + 2*X_1*T + 0*X_2*T + eps
    lambda X, T, eps: include_intercept  # intercept, non-causal => 0 no intercept
    + 1 * X["X_0"]  # non-causal
    - 1 * X["X_1"]  # non-causal
    + 1 * X["X_2"]  # non-causal
    + 5 * T  # causal
    + 2 * X["X_0"] * T  # causal
    + 2 * X["X_1"] * T  # causal
    - 4 * X["X_2"] * T  # causal
    + eps
)

CATE_function = lambda X: outcome_function(X,np.ones(len(X)), 0 )-outcome_function(X,np.zeros(len(X)),0)

# Prior parameters for Bayesian update on host
if include_intercept:
    prior_mean = np.array([0, 0, 0, 0, 0, 0, 0, 0])
else:
    prior_mean = np.array([0, 0, 0, 0, 0, 0, 0])
assert len(prior_mean) == d, "Shape error"

beta_0, sigma_0_sq, inv_cov_0 = (
    prior_mean,
    sigma_rand_error**2,
    1 / sigma_prior * np.eye(len(prior_mean)),
)
prior_hyperparameters = {
    "beta_0": beta_0,
    "sigma_0_sq": sigma_0_sq,
    "inv_cov_0": inv_cov_0,
}

n_seeds = 2
data_parameters = {
    "fixed_n_complementary": fixed_n_complementary,
    "varying_sample_sizes": varying_sample_sizes,
    "n_rct_before_split": n_rct_before_split,
    "x_distributions": x_distributions,
    "p_assigned_to_cand2": p_assigned_to_cand2,
    "p_assigned_to_host": p_assigned_to_host,
    "n_host": n_host,
    "power_x": power_x,
    "power_x_t": power_x_t,
    "outcome_function": outcome_function,
    "std_true_y": std_true_y,
    "causal_param_first_index": causal_param_first_index,
}

print('hyperparameters done')

def turn_into_diff(arr):
    n, d = np.shape(arr)[0], np.shape(arr)[1]
    result = np.zeros((n//2, d))
    for i in range (n//2):
        result[i,:]=arr[2*i,:]-arr[(2*i) +1,:]
    return result

proportions = np.array(varying_sample_sizes)/fixed_n_complementary
proportions

EIG_obs_closed_form_across_seeds, EIG_caus_closed_form_across_seeds = [], []
store_non_exact_data = {}

for i in range(n_seeds):
    nonexact_data = generate_data_varying_sample_size(
        data_parameters, include_intercept=bool(include_intercept), seed=i)
    EIGs = linear_eig_closed_form_varying_sample_size(  # CHECK what this does
        nonexact_data,
        data_parameters,
        prior_hyperparameters,
        verbose=False,
    )
    EIG_obs_closed_form_across_seeds.append(
        [cand_values for cand_values in EIGs[0].values()]
    )
    EIG_caus_closed_form_across_seeds.append(
        [cand_values for cand_values in EIGs[1].values()]
    )
    store_non_exact_data[i] = nonexact_data

print('computed closed form')

EIG_obs_closed_form_across_seeds = np.vstack(EIG_obs_closed_form_across_seeds)  
EIG_caus_closed_form_across_seeds = np.vstack(EIG_caus_closed_form_across_seeds)

predictive_closed_form = pd.DataFrame(turn_into_diff(EIG_obs_closed_form_across_seeds))
caus_closed_form = pd.DataFrame(turn_into_diff(EIG_caus_closed_form_across_seeds))

predictive_closed_form.to_csv("/home/ma/l/ltt19/code_causal_eig/data_results/predictive_closed_form.csv",mode='w+')
caus_closed_form.to_csv("/home/ma/l/ltt19/code_causal_eig/data_results/caus_closed_form.csv",mode='w+')

print('saved closed form')


n_samples_outer_expectation_obs = 400
n_samples_inner_expectation_obs = 800
n_samples_outer_expectation_caus = 400
n_samples_inner_expectation_caus = 800

sampling_parameters = {
    "n_samples_inner_expectation_obs": n_samples_inner_expectation_obs,
    "n_samples_outer_expectation_obs": n_samples_outer_expectation_obs,
    "n_samples_inner_expectation_caus": n_samples_inner_expectation_caus,
    "n_samples_outer_expectation_caus": n_samples_outer_expectation_caus,
}

EIG_obs_samples_across_seeds, EIG_caus_samples_across_seeds = [], []

for i in range(n_seeds):
    EIGs = linear_eig_from_samples_varying_sample_size(
        store_non_exact_data[i], data_parameters, prior_hyperparameters, sampling_parameters
    )
    EIG_obs_samples_across_seeds.append(
        [cand_values for cand_values in EIGs[0].values()]
    )
    EIG_caus_samples_across_seeds.append(
        [cand_values for cand_values in EIGs[1].values()]
    )
    

EIG_obs_samples_across_seeds = np.vstack(EIG_obs_samples_across_seeds)  
EIG_caus_samples_across_seeds = np.vstack(EIG_caus_samples_across_seeds)

print('computed mcmc samples')

predictive_mcmc = pd.DataFrame(turn_into_diff(EIG_obs_samples_across_seeds))
caus_mcmc = pd.DataFrame(turn_into_diff(EIG_caus_samples_across_seeds))

predictive_mcmc.to_csv("/home/ma/l/ltt19/code_causal_eig/data_results/predictive_mcmc.csv",mode='w+')
caus_mcmc.to_csv("/home/ma/l/ltt19/code_causal_eig/data_results/caus_mcmc.csv",mode='w+')

print('saved mcmc samples')


dict_all_diff = {'predictive mcmc': predictive_mcmc, 'causal mcmc': caus_mcmc, \
                 'predictive closed form': predictive_closed_form, 'causal closed form': caus_closed_form}
std_color_dict = {'predictive mcmc': 'orange', 'causal mcmc': 'plum', \
                 'predictive closed form': 'olivedrab', 'causal closed form': 'brown'}
mean_color_dict = {'predictive mcmc': 'chocolate', 'causal mcmc': 'purple', \
                 'predictive closed form': 'darkgreen', 'causal closed form': 'darkred'}


from scipy.interpolate import interp1d

def plot_dict(
    x,
    dict: dict,
    axis_names: list,
    mean_color_dict: dict = None,
    std_color_dict: dict = None,
    dict_additional_plots: Union [dict, None] = None,
    text: Union [str, None] = None,
    title: Union[str, None] = None,
    save: Union[str, None] = None,
    second_axis: Union[dict, None] = None,
):

    fig, ax1 = plt.subplots(figsize=(13, 10))

    for label, arr in dict.items():
        mean_data = np.mean(arr, axis=0)
        std_data = np.std(arr, axis=0)

        interp_mean_func = interp1d(x, mean_data, kind='linear')
        interp_std_func = interp1d(x, std_data, kind='linear')
        x_interp = np.linspace(min(x), max(x), num=len(mean_data)*3)  # Fine-grained x values for interpolation
        smooth_mean_data = interp_mean_func(x_interp)
        smooth_std_data = interp_std_func(x_interp)
        
        mean_color = mean_color_dict[label] if mean_color_dict is not None else "blue"
        std_color = std_color_dict[label] if std_color_dict is not None else "blue"

        ax1.plot(x_interp, smooth_mean_data, label=label, color=mean_color, linewidth=2.0)
        ax1.fill_between(
            x_interp, 
            smooth_mean_data - smooth_std_data, 
            smooth_mean_data + smooth_std_data, 
            color=std_color, alpha=0.25
            )
    if dict_additional_plots is not None:
        for key, arr in dict_additional_plots.items():
            ax1.plot(x, arr, label=key, linewidth=2.0)

    ax1.set_ylabel(axis_names[1], fontsize=23)
    ax1.set_xlabel(axis_names[0], fontsize=33)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.yaxis.set_label_coords(-0.085, 0.5)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(dict.keys())//2, fontsize=20)

    if second_axis:
        ax2 = ax1.twinx() 
        for label, arr in second_axis.items():
            ax2.plot(x, arr, label=label)
        ax2.set_ylabel(axis_names[2], fontsize=21, rotation=270, labelpad=15)
        ax2.tick_params(axis='y', labelsize=18)
        ax2.legend(loc='lower left', bbox_to_anchor=(0.10, 0.17), fontsize=20)
        ax2.yaxis.set_label_coords(1.13, 0.5)


    if title is not None:
        fig.suptitle(title)

    if text is not None:
        fig.text(
            0.5, -0.2, text, ha="center", va="center", transform=plt.gca().transAxes
        )

    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    fig.tight_layout()
    fig.gca().set_facecolor('#F5F5F5')

    if save:
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{save}_{current_date}_{current_time}.pdf"
        plt.savefig(filename, dpi=800)

    plt.show()

plot_dict(
    x = proportions,
    dict = dict_all_diff,
    mean_color_dict= mean_color_dict,
    std_color_dict = std_color_dict,
    axis_names=[r'$\frac{n_{twin}}{n_{complementary}}$', 'EIG(comp)-EIG(twin)', 'MSE(comp)-Mean MSE(twin)'],
    dict_additional_plots=None,
    text=None,
    title= None,
    second_axis={'post-merge CATE MSE':mean_mse['CATE']},
    save = "/home/ma/l/ltt19/code_causal_eig/plot_results/linear_synthetic.pdf"
)