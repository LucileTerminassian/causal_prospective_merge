"""
Description : Runs experiment on the twins dataset with simulated outcomes for selecting between multiple datasets

Usage: twins_ranking_linear.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
"""

import yaml
import numpy as np
import pandas as pd
import torch
from docopt import docopt
torch.set_default_tensor_type(torch.FloatTensor) 
import copy
import sys
import os
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import warnings
from tqdm import tqdm

notebook_dir = os.getcwd()
main_dir = notebook_dir[:notebook_dir.find("causal_prospective_merge")+ len("causal_prospective_merge")]
# Add the parent directory to the Python path
sys.path.append(main_dir )

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *
from research_exp_utils import *


args = docopt(__doc__)

# Load config file
with open(args['--cfg'], "r") as f:
    cfg = yaml.safe_load(f)

data_path = main_dir+"/"
data_with_groundtruth, x, t, y = get_data('twins', data_path)
data_with_groundtruth.dropna(inplace=True)
data_with_groundtruth = data_with_groundtruth.rename(columns={'t': 'T', 'y': 'Y'})
XandT = data_with_groundtruth.drop(columns=['Y','y0','y1','ite'])

number_of_candidate_sites = cfg["number_of_candidate_sites"]

min_sample_size_cand = cfg["min_sample_size_cand"]
max_sample_size_cand = cfg["max_sample_size_cand"]
host_sample_size = cfg["host_sample_size"]
desired_initial_sample_size = int(cfg["desired_initial_sample_size"])
print(desired_initial_sample_size)

k =cfg["k"]
top_n = cfg["top_n"]
print(top_n is None)
added_T_coef = cfg["number_of_candidate_sites"] # to increase importance of T

outcome_function = cfg["outcome_function"]
std_true_y = cfg["std_true_y"]
power_x = cfg["power_x"]
power_x_t = cfg["power_x_t"]
sigma_rand_error = cfg["sigma_rand_error"]
true_beta_great_0_prop = cfg["true_beta_great_0_prop"]

exp_parameters = {'number_of_candidate_sites': number_of_candidate_sites+1,
                'min_sample_size_cand': min_sample_size_cand, 
                'max_sample_size_cand': max_sample_size_cand, 
                'host_sample_size': host_sample_size, 
                'outcome_function': outcome_function, 
                'std_true_y': std_true_y, 
                'power_x': power_x, 
                'power_x_t': power_x_t}

causal_param_first_index = power_x*np.shape(XandT)[1]

correlation_with_true_rankings={}

# Create output directory if doesn't exists
now = datetime.now()
date_time_str = now.strftime("%m-%d %H:%M:%S")
direct_path = os.path.join(args['--o'],date_time_str)
os.makedirs(direct_path, exist_ok=True)
with open(os.path.join(direct_path, 'cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)
dump_path = os.path.join(direct_path, 'results.metrics.csv')

def generating_random_sites_from(data, exp_parameters, added_T_coef=1):
    
    candidates = {}
    sample_size, number_covariates = np.shape(data)[0], np.shape(data)[1]
    function_indices = {0: lambda X: np.log(X+1), 1: lambda X: X**3, 2: lambda X: X, 3: lambda X: X**2 }
    number_of_candidate_sites = exp_parameters['number_of_candidate_sites']
    min_sample_size_cand = exp_parameters['min_sample_size_cand']
    max_sample_size_cand = exp_parameters['max_sample_size_cand']
    outcome_function = None
    std_true_y = exp_parameters['std_true_y']
    power_x = exp_parameters['power_x']
    power_x_t = exp_parameters['power_x_t']
    number_features = number_covariates
    created_sites = 0
    
    while created_sites < number_of_candidate_sites:

        np.random.seed(np.random.randint(10000))
        
        selected_features_for_subsampling = np.random.randint(2, size = number_features) 
        # binary bool vector representing selection for being an input of the sampling function
        random_coefs = [np.random.uniform(-10, 10) for _ in range(number_features)] 
        random_fct_idx = [np.random.randint(0, len(function_indices.keys())) for _ in range(number_features)] 
        
        def p_assigned_to_site(X, T, eps):
            result = 0
            for j in range(number_features-1):
                result += selected_features_for_subsampling[j] * random_coefs[j] * function_indices[random_fct_idx[j]](X[j])
            # here i use added_T_coef * random_coefs to increase importance of T
            result +=  added_T_coef * random_coefs[-1] *  function_indices[random_fct_idx[-1]](T) #selected_features_for_subsampling[-1]
            return sigmoid(result + eps)
        
        sample_size = np.random.randint(min_sample_size_cand, max_sample_size_cand + 1)  # Add 1 to include max_sample_size_cand

        if created_sites==0:
            sample_size = exp_parameters['host_sample_size']
        design_data_cand = subsample_one_dataset(XandT, p_assigned_to_site, sample_size, power_x, power_x_t, outcome_function, std_true_y, seed=np.random.randint(10000))
        design_data_cand = design_data_cand.dropna()
        any_nan = design_data_cand.isna().any().any()
        if not design_data_cand.empty and not any_nan: # we're appending
            candidates[created_sites] = design_data_cand
        else:
            number_of_candidate_sites+=1 # not appending
        created_sites += 1

    return candidates

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in tqdm(range(cfg["num_repeats"])):
        XandT = XandT.sample(n=desired_initial_sample_size, replace=True, random_state=42)
        #dictionnary of random sites
        candidate_sites = generating_random_sites_from(XandT, exp_parameters, added_T_coef=50)

        for i, cand in candidate_sites.items():
            candidate_sites[i] = pd.concat([cand, data_with_groundtruth.loc[cand.index, 'Y']], axis=1)

        beta = (np.random.randn(152) > true_beta_great_0_prop)
        beta = beta - np.mean(beta)

        for i, cand in candidate_sites.items():
            candidate_sites[i]["Y"] = candidate_sites[i].drop(columns=["Y"]) @ beta 
            candidate_sites[i]["Y"] += std_true_y * np.random.randn(len(candidate_sites[i]["Y"]))

        host = candidate_sites[0]
        candidate_sites = {key: value for key, value in candidate_sites.items() if key != 0}
        XandT_host, Y_host = torch.from_numpy(host.drop(columns=["Y"]).values), torch.from_numpy(host["Y"].values)

        # Prior parameters for Bayesian update on host
        d = np.shape(host)[1]-1
        prior_mean = torch.zeros(d)
        beta_0, sigma_0_sq, inv_cov_0 = prior_mean, sigma_rand_error,torch.eye(d)
        prior_hyperparameters = {'beta_0': beta_0, 'sigma_0_sq': sigma_0_sq,"inv_cov_0":inv_cov_0}

        eig_results = {"EIG_obs_from_samples": [], 'EIG_caus_from_samples':[], "EIG_obs_closed_form":[], "EIG_caus_closed_form":[], "EIG_obs_bart":[], "EIG_caus_bart":[]}

        for _, candidate in candidate_sites.items():
            X_cand = torch.from_numpy(candidate.drop(columns=["Y"]).values)
            bayes_reg = BayesianLinearRegression(prior_hyperparameters)
            bayes_reg.set_causal_index(causal_param_first_index)
            post_host_parameters = bayes_reg.fit(XandT_host, Y_host)

            eig_results["EIG_obs_closed_form"].append(
                    bayes_reg.closed_form_obs_EIG(X_cand)
                    )
            eig_results["EIG_caus_closed_form"].append(
                    bayes_reg.closed_form_causal_EIG(X_cand)
                    )
            
        merged_datasets = {}

        for i, candidate in candidate_sites.items():
            merged_datasets[i]= pd.concat([host, candidate], axis=0)

        cate_diff = {}
        merged_mse = []
        XandT_host=host.drop(columns=["Y"])

        X_zero = XandT_host.copy() # we predict on host with T=0 and T=1
        X_zero.iloc[:,causal_param_first_index:] = 0

        X_one = XandT_host.copy()
        X_one.iloc[:,causal_param_first_index:] = XandT_host.iloc[:,:causal_param_first_index]

        merged_mse = []

        for i, candidate in merged_datasets.items():

            XandT_merged = candidate.drop(columns=["Y"])
            Y_merged = candidate['Y']

            learner = Ridge(fit_intercept=True)
            learner.fit(y=Y_merged, X=XandT_merged) # we fit on merged datasets

            true_cate = (X_one - X_zero) @ beta

            pred_cate = learner.predict(X_one)-learner.predict(X_zero)

            merged_mse.append(mean_squared_error(true_cate, pred_cate))

        obs_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_obs_closed_form"])), key=lambda i: eig_results["EIG_obs_closed_form"][i], reverse=True)

        caus_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_caus_closed_form"])), key=lambda i: eig_results["EIG_caus_closed_form"][i], reverse=True)


        # obs_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_obs_from_samples"])), key=lambda i: eig_results["EIG_obs_from_samples"][i], reverse=True)
        # print(obs_eig_ranking_from_samples)

        # caus_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_caus_from_samples"])), key=lambda i: eig_results["EIG_caus_from_samples"][i], reverse=True)
        # print(caus_eig_ranking_from_samples)

        true_cate_ranking = sorted(range(len(merged_mse)), key=lambda i: merged_mse[i], reverse=False) 



        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, obs_eig_ranking_closed_form, merged_mse=merged_mse, top_n = top_n, k = k)
        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, caus_eig_ranking_closed_form,merged_mse=merged_mse, top_n = top_n, k = k)

        ### random ranking
        random_ranking = np.random.choice(np.arange(1, number_of_candidate_sites+1), size=number_of_candidate_sites, replace=False)

        ### ranking by sample size
        sample_size_order = sorted(candidate_sites.keys(), key=lambda key: -candidate_sites[key].shape[0])

        ### ranking by similarity of covariate distribution
        mean_vector_host = XandT_host.iloc[:,:causal_param_first_index].mean()
        cov_matrix_host = XandT_host.iloc[:,:causal_param_first_index].cov()
        mvn = multivariate_normal(mean=mean_vector_host, cov=cov_matrix_host, allow_singular=1)
        # get log likelihood of candidate sites
        log_likelihood_list=[]
        for i, candidate in candidate_sites.items():
            log_likelihoods=mvn.logpdf(candidate.iloc[:,:causal_param_first_index].values)
            log_likelihood_list.append(np.mean(log_likelihoods))

        similarity_cov_distrib_ranking= sorted(range(len(log_likelihood_list)), key=lambda i: log_likelihood_list[i], reverse=True)

        ### ranking by similarity of propensity scores
        # we fit a propensity score model at target site and store logloss
        # for each site: we fit the model further on the cand site and compute log
        # nd assess the loss. Sites associated with loss values with higher discrepancy from the host should have distinct 
        #treatment allocation scheme, and thus be a better fit. 

        ps_model = LogisticRegression(fit_intercept=True)
        ps_model.fit(XandT_host.iloc[:,:causal_param_first_index], XandT_host['T'])
        t_host_pred = ps_model.predict(XandT_host.iloc[:,:causal_param_first_index])
        mse_host = mean_squared_error(t_host_pred, XandT_host['T'])
        mse_diff_list = []


        for i, candidate in candidate_sites.items():
            # ps_model_copy= copy.deepcopy(ps_model)
            # ps_model_copy.fit(candidate.iloc[:,:causal_param_first_index], candidate['T'])
            t_cand_pred = ps_model.predict(candidate.iloc[:,:causal_param_first_index]) # predict on host!
            mse_cand = abs(mean_squared_error(t_cand_pred, candidate['T']) - mse_host)
            mse_diff_list.append(mse_cand)

        similarity_pscore_ranking = sorted(range(len(mse_diff_list)), key=lambda i: mse_diff_list[i], reverse=True) 
        # the more diff in pscore the better so reverse=True

        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, list(random_ranking),merged_mse=merged_mse, top_n = top_n, k = k)
        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, list(sample_size_order),merged_mse=merged_mse, top_n = top_n, k = k)
        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, list(similarity_cov_distrib_ranking),merged_mse=merged_mse, top_n = top_n, k = k)
        compare_to_ground_truth(correlation_with_true_rankings, true_cate_ranking, list(similarity_pscore_ranking),merged_mse=merged_mse, top_n = top_n, k = k)
        correlation_with_true_rankings["Method"] = correlation_with_true_rankings.get("Method",[]) + ['obs_closed_form', 'caus_closed_form', 'random', 'sample size', 'similarity_cov_distrib_ranking', 'similarity_pscore_ranking size']

        with open(dump_path, 'w') as f:
            yaml.dump(correlation_with_true_rankings, f)