"""
Description : Runs experiment on the twins dataset with simulated outcomes for selecting between multiple datasets

Usage: twins_ranking.py  [options] --cfg=<path_to_config> --o=<output_dir>

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
sys.path.append(main_dir)

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
data_name = cfg["data_name"]
binary_outcome = cfg["binary_outcome"]
data_with_groundtruth, x, t, y = get_data(data_name, data_path)
XandT = pd.concat([x,t], axis=1)
#data_with_groundtruth.drop(columns=['Y','y0','y1','ite'])


number_of_candidate_sites = cfg["number_of_candidate_sites"]

min_sample_size_cand = cfg["min_sample_size_cand"]
max_sample_size_cand = cfg["max_sample_size_cand"]
host_sample_size = cfg["host_sample_size"]
host_test_size = cfg["host_test_size"]
desired_initial_sample_size = int(cfg["desired_initial_sample_size"])

k =cfg["k"]
top_n = cfg["top_n"]

added_T_coef = cfg["number_of_candidate_sites"] # to increase importance of T

outcome_function = cfg["outcome_function"]
std_true_y = cfg["std_true_y"]
power_x = cfg["model_param"]["Linear"]["power_x"]
power_x_t = cfg["model_param"]["Linear"]["power_x_t"]
sigma_rand_error = cfg["model_param"]["Linear"]["sigma_rand_error"]
true_beta_great_0_prop = cfg["true_beta_great_0_prop"]

exp_parameters = {'number_of_candidate_sites': number_of_candidate_sites+1,
                'min_sample_size_cand': min_sample_size_cand, 
                'max_sample_size_cand': max_sample_size_cand, 
                'host_sample_size': host_sample_size,
                'host_test_size': host_test_size,
                'outcome_function': outcome_function, 
                'std_true_y': std_true_y, 
                'power_x': power_x, 
                'power_x_t': power_x_t}

causal_param_first_index = power_x*np.shape(XandT)[1]

correlation_with_true_rankings={}

if cfg["model_param"]["Linear"]["run"]:
    correlation_with_true_rankings["lin"] = {}

if cfg["model_param"]["GP"]["run"]:
    correlation_with_true_rankings["GP"] = {}

if cfg["model_param"]["BART"]["run"]:
    correlation_with_true_rankings["BART"] = {}

# Create output directory if doesn't exists
now = datetime.now()
date_time_str = now.strftime("%m-%d %H:%M:%S")
direct_path = os.path.join(args['--o'],date_time_str.replace(" ","-"))
os.makedirs(direct_path, exist_ok=True)
with open(os.path.join(direct_path, 'cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)
dump_path = os.path.join(direct_path, 'results.metrics')



with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for i in tqdm(range(cfg["num_repeats"])):
        # try:
            XandT = XandT.sample(n=desired_initial_sample_size, replace=True, random_state=42)
            #dictionnary of random sites
            candidate_sites = generating_random_sites_from(XandT, data_with_groundtruth, exp_parameters, added_T_coef=50, binary_outcome=binary_outcome)


            if data_name=="twins":
                dimension = np.shape(candidate_sites[0])[1]-1
                beta = (np.random.randn(dimension) > true_beta_great_0_prop)
                beta = beta - np.mean(beta)

                for i, cand in candidate_sites.items():
                    candidate_sites[i]["Y"] = candidate_sites[i].drop(columns=["Y"]) @ beta 
                    candidate_sites[i]["Y"] += std_true_y * np.random.randn(len(candidate_sites[i]["Y"]))

            host = candidate_sites[0][:host_sample_size]
            XandT_host, Y_host = torch.from_numpy(host.drop(columns=["Y"]).values), torch.from_numpy(host["Y"].values)

            host_test = candidate_sites[0][host_sample_size:] # will be of size host_test_size
            XandT_host_test, Y_host_test = torch.from_numpy(host_test.drop(columns=["Y"]).values), torch.from_numpy(host_test["Y"].values)
            
            candidate_sites = {key: value for key, value in candidate_sites.items() if key != 0}

            # Prior parameters for Bayesian update on host
            d = np.shape(host)[1]-1
            prior_mean = torch.zeros(d)
            beta_0, sigma_0_sq, inv_cov_0 = prior_mean, sigma_rand_error,torch.eye(d)
            prior_hyperparameters = {'beta_0': beta_0, 'sigma_0_sq': sigma_0_sq,"inv_cov_0":inv_cov_0}

            eig_results = {"EIG_obs_GP": [], 'EIG_caus_GP':[], " EIG_obs_lin":[], " EIG_caus_lin":[], "EIG_obs_bart":[], "EIG_caus_bart":[]}

            
            if cfg["model_param"]["Linear"]["run"]:

                bayes_reg = BayesianLinearRegression(prior_hyperparameters)
                bayes_reg.set_causal_index(causal_param_first_index)
                post_host_parameters = bayes_reg.fit(XandT_host, Y_host)
            
            if cfg["model_param"]["GP"]["run"]:

                cGP = CausalGP()
                X_host = host.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values
                T_host = host["T"].values.astype(int)
                Y_host = host["Y"].values
                cGP.fit(X_host,T_host,Y_host)
            
            if cfg["model_param"]["BART"]["run"]:

                bcf = BayesianCausalForest(prior_hyperparameters=cfg["model_param"]["BART"]['prior_hyperparameters'],
                                           predictive_model_parameters = cfg["model_param"]["BART"]['predictive_model_parameters'],
                                           conditional_model_param =  cfg["model_param"]["BART"]['conditional_model_param'],
                                           )
                
                X_host = host.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values.astype(np.float32)
                T_host = host["T"].values.astype(np.int32).copy()
                Y_host = host["Y"].values
                bcf.store_train_data(X=X_host,T=T_host.astype(np.int32),Y=Y_host)
            
            XandT_host=host.drop(columns=["Y"])
            XandT_host_test = host_test.drop(columns=["Y"])

            for _, candidate in candidate_sites.items():

                X_cand = torch.from_numpy(candidate.drop(columns=["Y"]).values)

                if cfg["model_param"]["Linear"]["run"]:
                    eig_results[" EIG_obs_lin"].append(
                            bayes_reg.closed_form_obs_EIG(X_cand)
                            )
                    eig_results[" EIG_caus_lin"].append(
                            bayes_reg.closed_form_causal_EIG(X_cand)
                            )
                    
                if cfg["model_param"]["GP"]["run"]:
                    X_cand = candidate.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values
                    T_cand = candidate["T"].values.copy()
                    cGP.obs_EIG_closed_form(X_cand,T_cand)
                    eig_results["EIG_obs_GP"].append(
                            cGP.obs_EIG_closed_form(X_cand,T_cand)
                            )
                    eig_results["EIG_caus_GP"].append(
                            cGP.causal_EIG_closed_form(X_cand,T_cand)
                            )
                
                if cfg["model_param"]["BART"]["run"]:

                    X_cand = candidate.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values.astype(np.float32)
                    T_cand = candidate["T"].values.copy().astype(np.int32)
                    print(T_cand.dtype)
                    print(X_cand.dtype)
                    result = bcf.joint_EIG_calc_eff(X_cand,T_cand.astype(np.int32),sampling_parameters=cfg["model_param"]["BART"]['sampling_parameters'])
                    eig_results["EIG_obs_bart"].append(
                            result["Obs EIG"]
                            )
                    eig_results["EIG_caus_bart"].append(
                            result["Causal EIG"]
                            )
                                        

            merged_datasets = {}

            for i, candidate in candidate_sites.items():
                merged_datasets[i]= pd.concat([host, candidate], axis=0)

            cate_diff = {}

            XandT_host=host.drop(columns=["Y"])
            XandT_host_test = host_test.drop(columns=["Y"])

            X_zero = XandT_host_test.copy() # we predict on host with T=0 and T=1
            X_zero.iloc[:,causal_param_first_index:] = 0

            X_one = XandT_host_test.copy()
            X_one.iloc[:,causal_param_first_index:] = XandT_host_test.iloc[:,:causal_param_first_index]

            merged_mse_models = {}
            
            if data_name == "twins": # binary outcome
                true_cate = (X_one - X_zero) @ beta
            else: # use ground truth
                indices_list = XandT_host_test.index.tolist()
                true_cate = data_with_groundtruth['y1'].iloc[indices_list]- data_with_groundtruth['y0'].iloc[indices_list]

            for i, candidate in merged_datasets.items():
               

                if cfg["model_param"]["Linear"]["run"]:

                    XandT_merged = candidate.drop(columns=["Y"])
                    Y_merged = candidate['Y']

                    learner = Ridge(fit_intercept=True)
                    learner.fit(y=Y_merged, X=XandT_merged) # we fit on merged datasets
                    
                    
                    pred_cate = learner.predict(X_one)-learner.predict(X_zero)

                    merged_mse_models["lin"] = merged_mse_models.get("lin",[]) + [(mean_squared_error(true_cate, pred_cate))]

                    
                if cfg["model_param"]["BART"]["run"]:
                    
                    X_host_test = host_test.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values
                    
                    X_merged = candidate.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values.astype(np.float32)
                    T_merged = candidate["T"].values.astype(np.int32)
                    Y_merged = candidate["Y"].values.astype(np.float32)
        

                    bcf_new = BayesianCausalForest(prior_hyperparameters=cfg["model_param"]["BART"]['prior_hyperparameters'],
                                           predictive_model_parameters = cfg["model_param"]["BART"]['predictive_model_parameters'],
                                           conditional_model_param =  cfg["model_param"]["BART"]['conditional_model_param'],
                                           )
                    bcf_new.store_train_data(X=X_merged,Y=Y_merged,T=T_merged)
                    bcf_new.fit_model(1000)



                    pred_cate = bcf_new.model.predict(X_host_test)

                    if data_name == "twins": #binary outcome
                        true_cate = (X_one - X_zero) @ beta
                    else: # use ground truth
                        indices_list = XandT_host_test.index.tolist()
                        true_cate = data_with_groundtruth['y1'].iloc[indices_list]- data_with_groundtruth['y0'].iloc[indices_list]


                    merged_mse_models["BART"] = merged_mse_models.get("BART",[]) + [(mean_squared_error(true_cate, pred_cate))]
                
                if cfg["model_param"]["GP"]["run"]:
                    
                    X_host_test = host_test.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values
                    
                    X_merged = candidate.drop(columns=["Y","T"]).iloc[:,:causal_param_first_index].values
                    T_merged = candidate["T"].values.astype(int)
                    Y_merged = candidate["Y"].values
                    cGP_new = CausalGP()

                    cGP_new.fit(X_merged,T_merged,Y_merged)

                    if data_name == "twins": #binary outcome
                        true_cate = (X_one - X_zero) @ beta
                    elif data_name == "acic": # use ground truth
                        indices_list = XandT_host_test.index.tolist()
                        true_cate = data_with_groundtruth['y1'].iloc[indices_list]- data_with_groundtruth['y0'].iloc[indices_list]

                    pred_cate = cGP_new.pred_CATE(X_host_test)
                    merged_mse_models["GP"] = merged_mse_models.get("GP",[]) + [(mean_squared_error(true_cate, pred_cate))]
               
            if cfg["model_param"]["Linear"]["run"]:

                

                obs_eig_ranking_closed_form = sorted(range(len(eig_results[" EIG_obs_lin"])), key=lambda i: eig_results[" EIG_obs_lin"][i], reverse=True)

                caus_eig_ranking_closed_form = sorted(range(len(eig_results[" EIG_caus_lin"])), key=lambda i: eig_results[" EIG_caus_lin"][i], reverse=True)


                # obs_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_obs_from_samples"])), key=lambda i: eig_results["EIG_obs_from_samples"][i], reverse=True)
                # print(obs_eig_ranking_from_samples)

                # caus_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_caus_from_samples"])), key=lambda i: eig_results["EIG_caus_from_samples"][i], reverse=True)
                # print(caus_eig_ranking_from_samples)

                true_cate_ranking = sorted(range(len(merged_mse_models["lin"])), key=lambda i: merged_mse_models["lin"][i], reverse=False) 

                
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, obs_eig_ranking_closed_form, merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, caus_eig_ranking_closed_form,merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)

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

                
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, list(random_ranking),merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, list(sample_size_order),merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, list(similarity_cov_distrib_ranking),merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["lin"], true_cate_ranking, list(similarity_pscore_ranking),merged_mse=merged_mse_models["lin"], top_n = top_n, k = k)
                correlation_with_true_rankings["lin"]["Method"] = correlation_with_true_rankings["lin"].get("Method",[]) + ['obs_closed_form', 'caus_closed_form', 'random', 'sample size', 'similarity_cov_distrib_ranking', 'similarity_pscore_ranking size']


            if cfg["model_param"]["GP"]["run"]:


                obs_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_obs_GP"])), key=lambda i: eig_results["EIG_obs_GP"][i], reverse=True)

                caus_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_caus_GP"])), key=lambda i: eig_results["EIG_caus_GP"][i], reverse=True)


                # obs_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_obs_from_samples"])), key=lambda i: eig_results["EIG_obs_from_samples"][i], reverse=True)
                # print(obs_eig_ranking_from_samples)

                # caus_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_caus_from_samples"])), key=lambda i: eig_results["EIG_caus_from_samples"][i], reverse=True)
                # print(caus_eig_ranking_from_samples)

                true_cate_ranking = sorted(range(len(merged_mse_models["GP"])), key=lambda i: merged_mse_models["GP"][i], reverse=False) 


                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, obs_eig_ranking_closed_form, merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, caus_eig_ranking_closed_form,merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)

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

                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, list(random_ranking),merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, list(sample_size_order),merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, list(similarity_cov_distrib_ranking),merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["GP"], true_cate_ranking, list(similarity_pscore_ranking),merged_mse=merged_mse_models["GP"], top_n = top_n, k = k)
                correlation_with_true_rankings["GP"]["Method"] = correlation_with_true_rankings["GP"].get("Method",[]) + ['obs_closed_form', 'caus_closed_form', 'random', 'sample size', 'similarity_cov_distrib_ranking', 'similarity_pscore_ranking size']

            if cfg["model_param"]["BART"]["run"]:


                obs_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_obs_bart"])), key=lambda i: eig_results["EIG_obs_bart"][i], reverse=True)

                caus_eig_ranking_closed_form = sorted(range(len(eig_results["EIG_caus_bart"])), key=lambda i: eig_results["EIG_caus_bart"][i], reverse=True)


                # obs_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_obs_from_samples"])), key=lambda i: eig_results["EIG_obs_from_samples"][i], reverse=True)
                # print(obs_eig_ranking_from_samples)

                # caus_eig_ranking_from_samples = sorted(range(len(eig_results["EIG_caus_from_samples"])), key=lambda i: eig_results["EIG_caus_from_samples"][i], reverse=True)
                # print(caus_eig_ranking_from_samples)

                true_cate_ranking = sorted(range(len(merged_mse_models["BART"])), key=lambda i: merged_mse_models["BART"][i], reverse=False) 

                merged_mse_models["BART"]
                correlation_with_true_rankings["BART"]
                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, obs_eig_ranking_closed_form, merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, caus_eig_ranking_closed_form,merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)

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

                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, list(random_ranking),merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, list(sample_size_order),merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, list(similarity_cov_distrib_ranking),merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)
                compare_to_ground_truth(correlation_with_true_rankings["BART"], true_cate_ranking, list(similarity_pscore_ranking),merged_mse=merged_mse_models["BART"], top_n = top_n, k = k)
                correlation_with_true_rankings["BART"]["Method"] = correlation_with_true_rankings["BART"].get("Method",[]) + ['obs_closed_form', 'caus_closed_form', 'random', 'sample size', 'similarity_cov_distrib_ranking', 'similarity_pscore_ranking size']
            with open(dump_path, 'w') as f:
                yaml.dump(correlation_with_true_rankings, f)
                # except:
                #     print("Error, skipping to next")
                
            
for keys in correlation_with_true_rankings.keys():                 
    results_df = pd.DataFrame(correlation_with_true_rankings[keys])
    results_df.to_csv(os.path.join(direct_path, 'results'+keys+'.csv'))