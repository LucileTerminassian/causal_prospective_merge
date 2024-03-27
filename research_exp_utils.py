import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *

# data_parameters = {n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2,
# n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index}


def eig_closed_form_varying_sample_size(data_parameters, sigma_rand_error, prior_hyperparameters, n_mc):

# def eig_closed_form_varying_sample_size(n_both_candidates_list, proportion, n_rct_before_split, x_distributions, \
#                                         p_assigned_to_cand2, n_host, power_x, power_x_t, outcome_function, std_true_y, \
#                                         causal_param_first_index, sigma_rand_error, prior_hyperparameters, n_mc):
    
    n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2 = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['n_rct_before_split'], data_parameters['x_distributions'], data_parameters['p_assigned_to_cand2'], 
    n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index = data_parameters['n_host'], data_parameters['power_x'], \
            data_parameters['power_x_t'], data_parameters['outcome_function'], data_parameters['std_true_y'], data_parameters['causal_param_first_index']


    EIG_obs_closed_form_mirror, EIG_obs_closed_form_cand2 = [], []
    EIG_caus_closed_form_mirror, EIG_caus_closed_form_cand2 = [], []

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)


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
              
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']
        

        # X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        # mcmc_host = MCMC_Bayesian_update(X_torch = X_torch, Y_torch = Y_torch, model =model_normal,
        #             mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
        #             sigma_rand_error_fixed = True, n_mcmc = n_mc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

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


def eig_closed_form_exact_datasets(data_parameters, sigma_rand_error, prior_hyperparameters, n_mc):

    n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2 = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['n_rct_before_split'], data_parameters['x_distributions'], data_parameters['p_assigned_to_cand2'], 
    n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index = data_parameters['n_host'], data_parameters['power_x'], \
            data_parameters['power_x_t'], data_parameters['outcome_function'], data_parameters['std_true_y'], data_parameters['causal_param_first_index']


    EIG_obs_closed_form_exact_complementary, EIG_caus_closed_form_exact_complementary = [], []
    EIG_obs_closed_form_exact_twin, EIG_caus_closed_form_exact_twin = [], []
    EIG_obs_closed_form_exact_twin_treated, EIG_caus_closed_form_exact_twin_treated = [], []
    EIG_obs_closed_form_exact_twin_untreated, EIG_caus_closed_form_exact_twin_untreated = [], []

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)

        #initiate
        design_data_exact_complementary = design_data_exact_twin = design_data_exact_twin_untreated = design_data_exact_twin_treated = design_data_host.copy()

        complementary_treat = [1 if bit == 0 else 0 for bit in design_data_host['T']]
        design_data_exact_complementary['T'] = complementary_treat
        design_data_exact_twin_treated['T'] = 1
        design_data_exact_twin_untreated['T'] = 0

        print(design_data_exact_complementary.head())
        print(design_data_exact_twin.head())

        X_exact_complementary = design_data_exact_complementary.drop(columns=['Y'])
        X_exact_twin = design_data_exact_twin.drop(columns=['Y'])
        X_exact_twin_treated = design_data_exact_twin_treated.drop(columns=['Y'])
        X_exact_twin_untreated = design_data_exact_twin_untreated.drop(columns=['Y'])

        print(X_exact_complementary.head())
        print(X_exact_twin.head())

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']
        

        # X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)
        # mcmc_host = MCMC_Bayesian_update(X_torch = X_torch, Y_torch = Y_torch, model =model_normal,
        #             mu_0= beta_0, sigma_prior = sigma_prior, sigma_rand_error = sigma_rand_error,
        #             sigma_rand_error_fixed = True, n_mcmc = n_mc, warmup_steps = warmup_steps, max_tree_depth=max_tree_depth)

        X_exact_complementary_arr, X_exact_twin_arr = X_exact_complementary.values, X_exact_twin.values
        X_exact_twin_treated_arr, X_exact_twin_untreated_arr = X_exact_twin_treated.values, X_exact_twin_untreated.values

        EIG_obs_one_n_exact_complementary = compute_EIG_obs_closed_form(X_exact_complementary_arr, cov_matrix_post_host, sigma_rand_error)
        EIG_obs_one_n_exact_twin = compute_EIG_obs_closed_form(X_exact_twin_arr,cov_matrix_post_host,sigma_rand_error)
        EIG_obs_one_n_exact_treated = compute_EIG_obs_closed_form(X_exact_twin_treated_arr, cov_matrix_post_host, sigma_rand_error)
        EIG_obs_one_n_exact_untreated = compute_EIG_obs_closed_form(X_exact_twin_untreated_arr, cov_matrix_post_host, sigma_rand_error)

        EIG_obs_closed_form_exact_complementary.append(EIG_obs_one_n_exact_complementary)   
        EIG_obs_closed_form_exact_twin.append(EIG_obs_one_n_exact_twin) 
        EIG_obs_closed_form_exact_twin_treated.append(EIG_obs_one_n_exact_treated) 
        EIG_obs_closed_form_exact_twin_untreated.append(EIG_obs_one_n_exact_untreated)

        # print(X_host.head())
        # print(X_exact_complementary.head())
        # print(X_exact_twin.head())
        # print(X_exact_twin_treated.head())
        # print(X_exact_twin_untreated.head())

        EIG_caus_one_n_exact_complementary = compute_EIG_causal_closed_form(X_exact_complementary_arr, cov_matrix_post_host, sigma_rand_error, causal_param_first_index)
        EIG_caus_one_n_exact_twin = compute_EIG_causal_closed_form(X_exact_twin_arr,cov_matrix_post_host,sigma_rand_error, causal_param_first_index)
        EIG_caus_one_n_exact_treated = compute_EIG_causal_closed_form(X_exact_twin_treated_arr, cov_matrix_post_host, sigma_rand_error, causal_param_first_index)
        EIG_caus_one_n_exact_untreated = compute_EIG_causal_closed_form(X_exact_twin_untreated_arr, cov_matrix_post_host, sigma_rand_error, causal_param_first_index)

        EIG_caus_closed_form_exact_complementary.append(EIG_caus_one_n_exact_complementary)   
        EIG_caus_closed_form_exact_twin.append(EIG_caus_one_n_exact_twin) 
        EIG_caus_closed_form_exact_twin_treated.append(EIG_caus_one_n_exact_treated) 
        EIG_caus_closed_form_exact_twin_untreated.append(EIG_caus_one_n_exact_untreated)

    dict_results_obs={'Exact complementary': EIG_obs_closed_form_exact_complementary, 'Exact twin': EIG_obs_closed_form_exact_twin, \
                      'Exact twin treated': EIG_obs_closed_form_exact_twin_treated, 'Exact twin untreated': EIG_obs_closed_form_exact_twin_untreated}
    dict_results_caus={'Exact complementary': EIG_caus_closed_form_exact_complementary, 'Exact twin': EIG_caus_closed_form_exact_twin, \
                      'Exact twin treated': EIG_caus_closed_form_exact_twin_treated, 'Exact twin untreated': EIG_caus_closed_form_exact_twin_untreated}

    return dict_results_obs, dict_results_caus










# data_parameters = {n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2, n_host, power_x, 
# power_x_t, outcome_function, std_true_y, causal_param_first_index}
# sampling_parameters = {n_samples_for_expectation, m_samples_for_expectation, n_non_causal_expectation, n_causal_outer_exp}


# def eig_from_samples_varying_sample_size(n_both_candidates_list, proportion, n_rct_before_split, x_distributions, \
#                                         p_assigned_to_cand2, n_host, power_x, power_x_t, outcome_function, std_true_y, \
#                                         causal_param_first_index, sigma_rand_error, prior_hyperparameters, n_samples_for_expectation, m_samples_for_expectation, n_non_causal_expectation=100, n_causal_outer_exp=40):

def eig_from_samples_varying_sample_size(data_parameters, sigma_rand_error, prior_hyperparameters, sampling_parameters):

    EIG_obs_samples_mirror, EIG_obs_samples_cand2 = [], []
    EIG_caus_samples_mirror, EIG_caus_samples_cand2 = [], []

    n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2 = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['n_rct_before_split'], data_parameters['x_distributions'], data_parameters['p_assigned_to_cand2'], 
    n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index = data_parameters['n_host'], data_parameters['power_x'], \
            data_parameters['power_x_t'], data_parameters['outcome_function'], data_parameters['std_true_y'], data_parameters['causal_param_first_index']

    n_samples_for_expectation, m_samples_for_expectation, n_non_causal_expectation, n_causal_outer_exp = sampling_parameters['n_samples_for_expectation'],\
            sampling_parameters['m_samples_for_expectation'], sampling_parameters['n_non_causal_expectation'], sampling_parameters['n_causal_outer_exp'],

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)


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
                
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)

        ### Bayesian update on host data using closed form 
        X_host, Y_host= design_data_host.drop(columns=['Y']), design_data_host['Y']
        post_host_parameters = bayes_reg.fit(X_host, Y_host)
        beta_post_host_vec = post_host_parameters['posterior_mean'].flatten()  # Extract posterior mean
        cov_matrix_post_host = post_host_parameters['posterior_cov_matrix']

        n_samples = n_samples_for_expectation*(m_samples_for_expectation+1)
        beta_post_host_samples = bayes_reg.posterior_sample(n_samples=n_samples)
        
        X_mirror_arr, X_cand2_arr = X_mirror.values, X_cand2.values
        X_torch, Y_torch = torch.tensor(X_host.values), torch.tensor(Y_host.values)

        print('pred_obs')

        Y_pred_mirror = predict_with_all_sampled_linear(beta_post_host_samples, X_mirror_arr)
        pred_list_mirror = predictions_in_EIG_obs_form(Y_pred_mirror, n_samples_for_expectation, m_samples_for_expectation)   

        Y_pred_cand2 = predict_with_all_sampled_linear(beta_post_host_samples, X_cand2_arr)
        pred_list_cand2 = predictions_in_EIG_obs_form(Y_pred_cand2, n_samples_for_expectation, m_samples_for_expectation)  

        print('eig_obs')
        
        EIG_obs_one_n_mirror = compute_EIG_obs_from_samples(pred_list_mirror, sigma_rand_error)
        EIG_obs_one_n_cand2 = compute_EIG_obs_from_samples(pred_list_cand2, sigma_rand_error)

        EIG_obs_samples_mirror.append(EIG_obs_one_n_mirror)    
        EIG_obs_samples_cand2.append(EIG_obs_one_n_cand2)

        ## causal

        sample_func = bayes_reg.return_conditional_sample_function(causal_param_first_index)

        print('pred_caus')
        
        pred_func_mirror = lambda beta: beta @ (X_mirror).T
        pred_func_cand2 = lambda beta: beta @ (X_cand2).T

        pred_in_causal_form_mirror = predictions_in_EIG_causal_form(pred_func=pred_func_mirror, theta_samples=beta_post_host_samples[:n_causal_outer_exp], theta_sampling_function=sample_func,n_non_causal_expectation= n_non_causal_expectation,causal_param_first_index= causal_param_first_index)
        pred_in_causal_form_cand2 = predictions_in_EIG_causal_form(pred_func=pred_func_cand2, theta_samples=beta_post_host_samples[:n_causal_outer_exp], theta_sampling_function=sample_func,n_non_causal_expectation= n_non_causal_expectation,causal_param_first_index= causal_param_first_index)

        print('eig_caus')
        
        EIG_caus_one_n_mirror = compute_EIG_causal_from_samples(pred_list_mirror,pred_in_causal_form_mirror, sigma_rand_error)
        EIG_caus_one_n_cand2 = compute_EIG_causal_from_samples(pred_list_cand2,pred_in_causal_form_cand2, sigma_rand_error)

        EIG_caus_samples_mirror.append(EIG_caus_one_n_mirror)    
        EIG_caus_samples_cand2.append(EIG_caus_one_n_cand2)

    EIG_obs_samples = np.vstack((EIG_obs_samples_mirror, EIG_obs_samples_cand2 ))
    EIG_caus_samples = np.vstack((EIG_caus_samples_mirror, EIG_caus_samples_cand2 ))

    return EIG_obs_samples, EIG_caus_samples



