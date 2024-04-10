import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *



############################ CLOSED FORM

 ## noisy

def compute_eig_closed_form_varying_sample_size(data, data_parameters, sigma_rand_error, prior_hyperparameters, n_mc, verbose=True):
    
    n_both_candidates_list, proportion = data_parameters['n_both_candidates_list'], data_parameters['proportion']
    std_true_y, causal_param_first_index = data_parameters['std_true_y'], data_parameters['causal_param_first_index']
    
    results = {'EIG_obs_closed_form_mirror':[], 'EIG_obs_closed_form_cand2':[], 'EIG_caus_closed_form_mirror':[], 'EIG_caus_closed_form_cand2':[]}

    for length in n_both_candidates_list:

        n_both_candidates_list, proportion, causal_param_first_index = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['causal_param_first_index']

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= torch.from_numpy(data[length]['host'].drop(columns=['Y']).values), torch.from_numpy(data[length]['host']['Y'].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        # beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        # cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']
        # X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        # mcmc_host = MCMC_Bayesian_update(X_torch = X_torch, Y_torch = Y_torch, model =model_normal,
        #             mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
        #             sigma_rand_error_fixed = True, n_mcmc = n_mc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

        X_mirror = torch.from_numpy(data[length]['mirror'].drop(columns=['Y']).values)
        X_cand2 = torch.from_numpy(data[length]['cand2'].drop(columns=['Y']).values)
  
        # if verbose:
            
        #     percent_treated_host = 100*sum(X_host["T"])/len(X_host)
        #     percent_treated_mirror = 100*sum(X_mirror["T"])/len(X_mirror)
        #     percent_treated_cand2 = 100*sum(X_cand2["T"])/len(X_cand2)

        #     print("For a sample size in mirror and host of "+str(length))
        #     print(f'Percentage of treated in host: {percent_treated_host}%')
        #     print(f'Percentage of treated in mirror: {percent_treated_mirror}%')
        #     print(f'Percentage of treated in cand2: {percent_treated_cand2}%') 

        results['EIG_obs_closed_form_mirror'].append(bayes_reg.closed_form_obs_EIG(X_mirror))   
        results['EIG_obs_closed_form_cand2'].append(bayes_reg.closed_form_obs_EIG(X_cand2))   

        results['EIG_caus_closed_form_mirror'].append(bayes_reg.closed_form_causal_EIG(X_mirror))
        results['EIG_caus_closed_form_cand2'].append(bayes_reg.closed_form_causal_EIG(X_cand2))
    
    EIG_obs_closed_form = np.vstack((results['EIG_obs_closed_form_mirror'], results['EIG_obs_closed_form_cand2'] ))
    EIG_caus_closed_form = np.vstack((results['EIG_caus_closed_form_mirror'], results['EIG_caus_closed_form_cand2'] ))

    return EIG_obs_closed_form, EIG_caus_closed_form

 ## exact


def compute_eig_closed_form_exact_datasets(data, data_parameters, sigma_rand_error, prior_hyperparameters, n_mc):

    n_both_candidates_list, proportion = data_parameters['n_both_candidates_list'], data_parameters['proportion']
    std_true_y, causal_param_first_index = data_parameters['std_true_y'], data_parameters['causal_param_first_index']


    dict_results_obs={'complementary': [], 'twin': [], \
                      'twin_treated': [], 'twin_untreated': []}
    dict_results_caus={'complementary': [], 'twin': [], \
                      'twin_treated': [], 'twin_untreated': []}

    for length in n_both_candidates_list:

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= torch.from_numpy(data[length]['host'].drop(columns=['Y']).values), torch.from_numpy(data[length]['host']['Y'].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        # beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        # cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']
        # X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        # mcmc_host = MCMC_Bayesian_update(X_torch = X_torch, Y_torch = Y_torch, model =model_normal,
        #             mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
        #             sigma_rand_error_fixed = True, n_mcmc = n_mc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)
        # bayes_reg.closed_form_obs_EIG / bayes_reg.closed_form_causal_EIG

        X_exact_complementary = torch.from_numpy(data[length]['exact_complementary'].drop(columns=['Y']).values)
        X_exact_twin = torch.from_numpy(data[length]['exact_twin'].drop(columns=['Y']).values)
        X_exact_twin_treated = torch.from_numpy(data[length]['exact_twin_treated'].drop(columns=['Y']).values)
        X_exact_twin_untreated = torch.from_numpy(data[length]['exact_twin_untreated'].drop(columns=['Y']).values)

        dict_results_obs['complementary'].append(bayes_reg.closed_form_obs_EIG(X_exact_complementary))
        dict_results_obs['twin'].append(bayes_reg.closed_form_obs_EIG(X_exact_twin))
        dict_results_obs['twin_treated'].append(bayes_reg.closed_form_obs_EIG(X_exact_twin_treated))
        dict_results_obs['twin_untreated'].append(bayes_reg.closed_form_obs_EIG(X_exact_twin_untreated))

        dict_results_caus['complementary'].append(bayes_reg.closed_form_causal_EIG(X_exact_complementary))
        dict_results_caus['twin'].append(bayes_reg.closed_form_causal_EIG(X_exact_twin))
        dict_results_caus['twin_treated'].append(bayes_reg.closed_form_causal_EIG(X_exact_twin_treated))
        dict_results_caus['twin_untreated'].append(bayes_reg.closed_form_causal_EIG(X_exact_twin_untreated))
        
    return dict_results_obs, dict_results_caus


#################################### FROM SAMPLES


## noisy

def compute_eig_from_samples_varying_sample_size(data, data_parameters, prior_hyperparameters, sampling_parameters):

    results = {'EIG_obs_from_samples_mirror':[], 'EIG_obs_from_samples_cand2':[], 'EIG_caus_from_samples_mirror':[], 'EIG_caus_from_samples_cand2':[]}

    n_both_candidates_list, proportion, = data_parameters['n_both_candidates_list'],  data_parameters['proportion']
    causal_param_first_index = data_parameters['causal_param_first_index']

    n_samples_outer_expectation, n_samples_inner_expectation, n_causal_inner_exp, n_causal_outer_exp = sampling_parameters['n_samples_outer_expectation'],\
        sampling_parameters['n_samples_inner_expectation'], sampling_parameters['n_causal_inner_exp'], sampling_parameters['n_causal_outer_exp']

    for length in n_both_candidates_list:

        X_host, Y_host= torch.from_numpy(data[length]['host'].drop(columns=['Y']).values), torch.from_numpy(data[length]['host']['Y'].values)
        X_mirror = torch.from_numpy(data[length]['mirror'].drop(columns=['Y']).values)
        X_cand2 = torch.from_numpy(data[length]['cand2'].drop(columns=['Y']).values)

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        # beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        # cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']

        n_samples = n_samples_outer_expectation*(n_samples_inner_expectation+1)
        beta_post_host_samples = bayes_reg.posterior_sample(n_samples=n_samples)
        
        results['EIG_obs_from_samples_mirror'].append(bayes_reg.samples_obs_EIG(X_mirror, n_samples_outer_expectation, n_samples_inner_expectation))   
        results['EIG_obs_from_samples_cand2'].append(bayes_reg.samples_obs_EIG(X_cand2, n_samples_outer_expectation, n_samples_inner_expectation))   

        results['EIG_caus_from_samples_mirror'].append(bayes_reg.samples_causal_EIG(X_mirror, n_causal_outer_exp, n_causal_inner_exp))
        results['EIG_caus_from_samples_cand2'].append(bayes_reg.samples_causal_EIG(X_cand2, n_causal_outer_exp, n_causal_inner_exp))
    
    EIG_obs_samples = np.vstack((results['EIG_obs_from_samples_mirror'], results['EIG_obs_from_samples_cand2']))
    EIG_caus_samples = np.vstack((results['EIG_caus_from_samples_mirror'], results['EIG_caus_from_samples_cand2']))

    return EIG_obs_samples, EIG_caus_samples



 ## exact


def compute_eig_from_samples_exact_datasets(data, data_parameters, prior_hyperparameters, n_mc, sampling_parameters):

    n_both_candidates_list, proportion, = data_parameters['n_both_candidates_list'],  data_parameters['proportion']
    causal_param_first_index = data_parameters['causal_param_first_index']

    n_samples_outer_expectation, n_samples_inner_expectation, n_causal_inner_exp, n_causal_outer_exp = sampling_parameters['n_samples_outer_expectation'],\
    sampling_parameters['n_samples_inner_expectation'], sampling_parameters['n_causal_inner_exp'], sampling_parameters['n_causal_outer_exp']


    dict_results_obs={'complementary': [], 'twin': [], \
                      'twin_treated': [], 'twin_untreated': []}
    dict_results_caus={'complementary': [], 'twin': [], \
                      'twin_treated': [], 'twin_untreated': []}

    for length in n_both_candidates_list:

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= torch.from_numpy(data[length]['host'].drop(columns=['Y']).values), torch.from_numpy(data[length]['host']['Y'].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        n_samples = n_samples_outer_expectation*(n_samples_inner_expectation+1)
        beta_post_host_samples = bayes_reg.posterior_sample(n_samples=n_samples)

        X_exact_complementary = torch.from_numpy(data[length]['exact_complementary'].drop(columns=['Y']).values)
        X_exact_twin = torch.from_numpy(data[length]['exact_twin'].drop(columns=['Y']).values)
        X_exact_twin_treated = torch.from_numpy(data[length]['exact_twin_treated'].drop(columns=['Y']).values)
        X_exact_twin_untreated = torch.from_numpy(data[length]['exact_twin_untreated'].drop(columns=['Y']).values)

        dict_results_obs['complementary'].append(bayes_reg.samples_obs_EIG(X_exact_complementary, n_samples_outer_expectation, n_samples_inner_expectation))   
        dict_results_obs['twin'].append(bayes_reg.samples_obs_EIG(X_exact_twin, n_samples_outer_expectation, n_samples_inner_expectation))   
        dict_results_obs['twin_treated'].append(bayes_reg.samples_obs_EIG(X_exact_twin_treated, n_samples_outer_expectation, n_samples_inner_expectation))   
        dict_results_obs['twin_untreated'].append(bayes_reg.samples_obs_EIG(X_exact_twin_untreated, n_samples_outer_expectation, n_samples_inner_expectation))   

        dict_results_caus['complementary'].append(bayes_reg.samples_causal_EIG(X_exact_complementary, n_causal_outer_exp, n_causal_inner_exp))   
        dict_results_caus['twin'].append(bayes_reg.samples_causal_EIG(X_exact_twin, n_causal_outer_exp, n_causal_inner_exp))   
        dict_results_caus['twin_treated'].append(bayes_reg.samples_causal_EIG(X_exact_twin_treated, n_causal_outer_exp, n_causal_inner_exp))   
        dict_results_caus['twin_untreated'].append(bayes_reg.samples_causal_EIG(X_exact_twin_untreated, n_causal_outer_exp, n_causal_inner_exp)) 
        
    return dict_results_obs, dict_results_caus
