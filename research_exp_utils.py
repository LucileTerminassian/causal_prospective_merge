import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *

def eig_closed_form_varying_sample_size(host_mirror_generation_function, n_both_candidates_list, proportion, n_rct_before_split, x_distributions, \
                                         p_assigned_to_cand2, n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index, \
                                        max_tree_depth, warmup_steps, sigma_rand_error, sigma_prior):

    EIG_obs_closed_form_mirror, EIG_obs_closed_form_cand2 = [], []
    EIG_caus_closed_form_mirror, EIG_caus_closed_form_cand2 = [], []

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = host_mirror_generation_function(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)


        pre_X_cand2, pre_T_cand2 = generate_rct(n_rct_before_split, x_distributions)
        design_data_cand2 = generate_cand2(pre_X_cand2, pre_T_cand2, p_assigned_to_cand2, proportion*length, \
                                            power_x, power_x_t, outcome_function, std_true_y)

        X_host = design_data_host.drop(columns=['Y'])
        X_mirror = design_data_mirror.drop(columns=['Y'])
        X_cand2 = design_data_cand2.drop(columns=['Y'])

        percent_treated_host = 100*sum(X_host["T"])/len(X_host)
        percent_treated_mirror = 100*sum(X_mirror["T"])/len(X_mirror)
        percent_treated_cand2 = 100*sum(X_cand2["T"])/len(X_cand2)

        print("For a sample size in mirror and host of "+str(length))
        print(f'Percentage of treated in host: {percent_treated_host}%')
        print(f'Percentage of treated in mirror: {percent_treated_mirror}%')
        print(f'Percentage of treated in cand2: {percent_treated_cand2}%')
              
        # Initialize prior parameters
        beta_0, sigma_0_sq = prior_mean, 1
        prior_hyperparameters = {'beta_0': beta_0, 'sigma_0_sq': sigma_0_sq}
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)

        ### Bayesian update through host dataset
        X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']

        ## Bayesian update using the host dataset
        X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        mcmc_host = MCMC_Bayesian_update(X_torch =X_torch, Y_torch = Y_torch, model =model_normal,
                    mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
                    sigma_rand_error_fixed = True, n_mcmc = n_mcmc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

        X_mirror_arr, X_cand2_arr = X_mirror.values, X_cand2.values

        EIG_obs_one_n_mirror = compute_EIG_obs_closed_form(X_mirror_arr,cov_matrix_post_host,sigma_rand_error)
        EIG_obs_one_n_cand2 = compute_EIG_obs_closed_form(X_cand2_arr, cov_matrix_post_host, sigma_rand_error)

        EIG_obs_closed_form_mirror.append(EIG_obs_one_n_mirror)    
        EIG_obs_closed_form_cand2.append(EIG_obs_one_n_cand2)

        EIG_caus_one_n_mirror = compute_EIG_causal_closed_form(X_mirror_arr, cov_matrix_post_host, sigma_rand_error, causal_param_first_index)
        EIG_caus_one_n_cand2 = compute_EIG_causal_closed_form(X_cand2_arr, cov_matrix_post_host, sigma_rand_error, causal_param_first_index)

        EIG_caus_closed_form_mirror.append(EIG_caus_one_n_mirror)    
        EIG_caus_closed_form_cand2.append(EIG_caus_one_n_cand2)
    
    EIG_obs_closed_form = np.vstack((EIG_obs_closed_form_mirror,EIG_obs_closed_form_cand2 ))
    EIG_caus_closed_form = np.vstack((EIG_caus_closed_form_mirror,EIG_caus_closed_form_cand2 ))

    return EIG_obs_closed_form, EIG_caus_closed_form



def eig_from_samples_varying_sample_size(n_both_candidates_list, n_rct_before_split, x_distributions, p_assigned_to_cand2, n_host, \
                                         power_x, power_x_t, outcome_function, std_true_y, n_non_causal_expectation, causal_param_first_index, \
                                        max_tree_depth, warmup_steps, sigma_rand_error, prior_mean, sigma_prior):

    EIG_obs_samples_mirror, EIG_obs_samples_cand2 = [], []
    EIG_caus_samples_mirror, EIG_caus_samples_cand2 = [], []

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, \
                                             n_host, length, power_x, power_x_t, outcome_function, std_true_y)


        pre_X_cand2, pre_T_cand2 = generate_rct(n_rct_before_split, x_distributions)
        design_data_cand2 = generate_cand2(pre_X_cand2, pre_T_cand2, p_assigned_to_cand2, length, \
                                            power_x, power_x_t, outcome_function, std_true_y)
        
        X_mirror = design_data_mirror.drop(columns=['Y'])
        X_cand2 = design_data_cand2.drop(columns=['Y'])
        print(np.shape(X_mirror))
        
        # Initialize prior parameters
        beta_0, sigma_0_sq = prior_mean, 1
        prior_hyperparameters = {'beta_0': beta_0, 'sigma_0_sq': sigma_0_sq}
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)

        ### Bayesian update through host dataset
        X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']

        ## Bayesian update using the host dataset
        X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        mcmc_host = MCMC_Bayesian_update(X_torch =X_torch, Y_torch = Y_torch, model = model_normal,
                    mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
                    sigma_rand_error_fixed = True, n_mcmc = n_mcmc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

        beta_post_host = pd.DataFrame(mcmc_host.get_samples())
        X_mirror_arr, X_cand2_arr = X_mirror.values, X_cand2.values

        ## obs

        Y_pred_mirror = predict_with_all_sampled_linear(beta_post_host, X_mirror_arr)
        pred_list_mirror = predictions_in_EIG_obs_form(Y_pred_mirror, n_samples_for_expectation, m_samples_for_expectation)   

        Y_pred_cand2 = predict_with_all_sampled_linear(beta_post_host, X_cand2_arr)
        pred_list_cand2 = predictions_in_EIG_obs_form(Y_pred_cand2, n_samples_for_expectation, m_samples_for_expectation)  
        
        EIG_obs_one_n_mirror = compute_EIG_obs_from_samples(pred_list_mirror, sigma_rand_error)
        EIG_obs_one_n_cand2 = compute_EIG_obs_from_samples(pred_list_cand2, sigma_rand_error)

        EIG_obs_samples_mirror.append(EIG_obs_one_n_mirror)    
        EIG_obs_samples_cand2.append(EIG_obs_one_n_cand2)

        ## causal

        sample_func = return_causal_samp_func_linear(X=X_torch,Y=Y_torch,causal_param_first_index=3,mu_0=beta_0,sigma_prior = sigma_prior,
                                            sigma_rand_error = sigma_rand_error,sigma_rand_error_fixed = True,warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)
        
        pred_func = lambda beta: beta @ (X_mirror).T
        
        pred_in_causal_form_mirror = predictions_in_EIG_causal_form(pred_func, n_non_causal_expectation, causal_param_first_index, theta_samples=beta_post_host.values[:40], theta_sampling_function=sample_func)
        pred_in_causal_form_cand2 = predictions_in_EIG_causal_form(pred_func, n_non_causal_expectation, causal_param_first_index, theta_samples=beta_post_host.values[:40], theta_sampling_function=sample_func)


        EIG_caus_one_n_mirror = compute_EIG_causal_from_samples(pred_list_mirror,pred_in_causal_form_mirror, sigma_rand_error)
        EIG_caus_one_n_cand2 = compute_EIG_causal_from_samples(pred_list_cand2,pred_in_causal_form_cand2, sigma_rand_error)

        EIG_caus_samples_mirror.append(EIG_caus_one_n_mirror)    
        EIG_caus_samples_cand2.append(EIG_caus_one_n_cand2)

    EIG_obs_samples = np.vstack((EIG_obs_samples_mirror, EIG_obs_samples_cand2 ))
    EIG_caus_samples = np.vstack((EIG_caus_samples_mirror, EIG_caus_samples_cand2 ))

    return EIG_obs_samples, EIG_caus_samples
