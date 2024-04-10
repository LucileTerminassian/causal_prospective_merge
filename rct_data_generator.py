import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
import sys

def generate_rct(n_global, x_distributions):
    # Generate X
    dim_x = len(x_distributions)
    dict_x = {}
    for name, x_column in x_distributions.items():
        dict_x[name] = x_column
    # for i in range(dim_x):
        # x = x_distributions[i]
        # dict_x[f"X{i}"] = x
    X = pd.DataFrame.from_dict(dict_x)
    # Generate T
    T = np.random.randint(0, 2, size=n_global)
    
    return X, T

# Probability functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to generate host and mirror data
def generate_host_and_mirror(X, T, f_assigned_to_host, n_host, n_mirror, power_x, power_x_t, outcome_function, std_true_y):

    n_global = np.shape(X)[0]
    # Initialize dictionaries
    data_host, data_mirror = {}, {}
    
    # Add 'T' key to each dictionary
    for name in X.columns:
        data_host[name] = []
        data_mirror[name] = []
    # Add 'T' key to each dictionary
    data_host['T'] = []
    data_mirror['T'] = []

    if n_host + n_mirror > n_global:
        print('n_host + n_mirror > n_rct')
        return 
    
    done_mirror, done_host = False, False
    
    for i in range(n_global):

        proba_assigned_to_host = f_assigned_to_host(X.iloc[i,:], T[i], np.random.normal())
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)

        first_column_host = next(iter(data_host.values()))
        first_column_mirror = next(iter(data_mirror.values()))

        if is_assigned_to_host:
            if len(first_column_host) < n_host :
                for column_name in X.columns:
                    data_host[column_name].append(X.iloc[i][column_name])
                data_host['T'].append(T[i])
            else:
                done_host = True
                pass

        else:
            if len(first_column_mirror) < n_mirror :
                for column_name in X.columns:
                    data_mirror[column_name].append(X.iloc[i][column_name])
                data_mirror['T'].append(T[i])
            else:
                done_mirror = True
                pass
        if done_mirror and done_host:
            break


    data_host = pd.DataFrame.from_dict(data_host)
    data_mirror = pd.DataFrame.from_dict(data_mirror)

    design_data_host = generate_design_matrix(data_host, power_x, power_x_t)
    design_data_mirror = generate_design_matrix(data_mirror, power_x, power_x_t)

    design_data_host = add_outcome(design_data_host, outcome_function, std_true_y)
    design_data_mirror = add_outcome(design_data_mirror, outcome_function, std_true_y)
    
    return design_data_host, design_data_mirror



# Function to generate host2 data
def generate_cand2(X, T, f_assigned_to_cand2, n_cand2, power_x, power_x_t, outcome_function, std_true_y):
   
    n_global = np.shape(X)[0]
    # Initialize dictionaries
    data_cand2 = {}
    for name in X.columns:
        data_cand2[name] = []

    # Add 'T' key to each dictionary
    data_cand2['T'] = []


    if n_cand2 > n_global:
        print('n_cand2 > n_rct')
        return

    for i in range(n_global):
        proba_assigned_to_cand2 = f_assigned_to_cand2(X.iloc[i,:], T[i], np.random.normal())
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 == 1:
            first_value = next(iter(data_cand2.values()))
            if len(first_value) < n_cand2 :
                for column_name in X.columns:
                    data_cand2[column_name].append(X.iloc[i][column_name])
                data_cand2['T'].append(T[i])
            else:
                break
    
    data_cand2 = pd.DataFrame.from_dict(data_cand2)

    design_cand2 = generate_design_matrix(data_cand2, power_x, power_x_t)
    design_cand2 = add_outcome(design_cand2, outcome_function, std_true_y)
    
    return design_cand2

def generate_design_matrix(data, power_x, power_x_t):
    # Extract X and T from the dataframe
    X = data.drop(columns=['T'])
    T = data['T']

    # Initialize a dataframe to hold the design matrix with intercept column filled with ones
    n, d = np.shape(X)
    X_prime = pd.DataFrame(np.ones((n, d*power_x + d*power_x_t + 2)))

    # Create a list to hold column names
    column_names = ['intercept']

    for i in range(1, power_x + 1):
        for col in X.columns:
            if i>1:
                column_names.append(f"{col}**{i}")
            else:
                column_names.append(f"{col}")

    column_names.append("T")

    for i in range(1, power_x_t + 1):
        for col in X.columns:
            if i>1:
                column_names.append(f"T*{col}**{i}")
            else:
                column_names.append(f"T*{col}")
    
    # Set column names for X_prime
    X_prime.columns = column_names

    # Concatenate X^i for i = 1 to power_x
    for i in range(1, power_x + 1):
        for col in X.columns:
            if i>1:
                X_prime[f"{col}**{i}"] = X[col] ** i
            else:
                X_prime[f"{col}"] = X[col] ** i

    X_prime["T"] = T 

    # Concatenate T*X^i for i = 1 to power_x_t
    for i in range(1, power_x_t + 1):
        for col in X.columns:
            if i>1:
                X_prime[f"T*{col}**{i}"] = T * (X[col] ** i)
            else:
                X_prime[f"T*{col}"] = T * (X[col] ** i)

    return X_prime

def add_outcome(data, outcome_function, scale):
    
    n = np.shape(data)[0]
    X = data.drop(columns=['T'])
    T = data['T']
    eps = np.random.normal(size=n, scale=scale)

    Y = outcome_function(X, T, eps)
    data['Y'] = Y

    return data

def generate_synthetic_data_varying_sample_size(data_parameters, print=True):

    
    n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2 = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['n_rct_before_split'], data_parameters['x_distributions'], data_parameters['p_assigned_to_cand2'], 
    n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index = data_parameters['n_host'], data_parameters['power_x'], \
            data_parameters['power_x_t'], data_parameters['outcome_function'], data_parameters['std_true_y'], data_parameters['causal_param_first_index']
    
    data = {}

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)

        pre_X_cand2, pre_T_cand2 = generate_rct(n_rct_before_split, x_distributions)
        design_data_cand2 = generate_cand2(pre_X_cand2, pre_T_cand2, p_assigned_to_cand2, proportion*length, \
                                            power_x, power_x_t, outcome_function, std_true_y)
       

        data[length] = {'host': design_data_host, 'mirror': design_data_mirror, 'cand2': design_data_cand2}

    return data

def generate_exact_synthetic_data_varying_sample_size(data_parameters):

    n_both_candidates_list, proportion, n_rct_before_split, x_distributions, p_assigned_to_cand2 = data_parameters['n_both_candidates_list'], \
            data_parameters['proportion'], data_parameters['n_rct_before_split'], data_parameters['x_distributions'], data_parameters['p_assigned_to_cand2'], 
    n_host, power_x, power_x_t, outcome_function, std_true_y, causal_param_first_index = data_parameters['n_host'], data_parameters['power_x'], \
            data_parameters['power_x_t'], data_parameters['outcome_function'], data_parameters['std_true_y'], data_parameters['causal_param_first_index']
    
    data = {}

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(n_rct_before_split, x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(X_rct, T_rct, p_assigned_to_cand2, n_host, length, power_x, power_x_t, outcome_function, std_true_y)
        number_x_features = 1 + np.shape(X_rct)[1]
        X_host = design_data_host.iloc[:, :number_x_features]

        #exact_complementary
        complementary_treat = pd.DataFrame([1 if bit == 0 else 0 for bit in design_data_host['T']], columns=['T'])
        data_complementary = pd.concat([X_host.iloc[:,1:], complementary_treat], axis=1)
        design_data_exact_complementary = generate_design_matrix(data_complementary, power_x, power_x_t)
        design_data_exact_complementary = add_outcome(design_data_exact_complementary, outcome_function, std_true_y)

        #exact_twin
        design_data_exact_twin = design_data_host.copy()

        #exact_twin_untreated
        untreated =  pd.DataFrame([0] * len(complementary_treat), columns=['T'])
        data_exact_twin_untreated = pd.concat([X_host.iloc[:,1:], untreated], axis=1)
        design_data_exact_twin_untreated = generate_design_matrix(data_exact_twin_untreated, power_x, power_x_t)
        design_data_exact_twin_untreated = add_outcome(design_data_exact_twin_untreated, outcome_function, std_true_y)

        #exact_twin_treated
        treated =  pd.DataFrame([1] * len(complementary_treat), columns=['T'])
        data_exact_twin_treated = pd.concat([X_host.iloc[:,1:], treated], axis=1)
        design_data_exact_twin_treated = generate_design_matrix(data_exact_twin_treated, power_x, power_x_t)
        design_data_exact_twin_treated = add_outcome(design_data_exact_twin_treated, outcome_function, std_true_y)


        ### if needed, expansion

        num_samples_needed = length - len(X_host)
        if num_samples_needed > 0:
             
             sampled_data_complementary = design_data_exact_complementary.sample(n=num_samples_needed, replace=True)
             design_data_exact_complementary = pd.concat([design_data_exact_complementary, sampled_data_complementary], ignore_index=True)

             sampled_data_twin = design_data_exact_twin.sample(n=num_samples_needed, replace=True)
             design_data_exact_twin = pd.concat([design_data_exact_twin, sampled_data_twin], ignore_index=True)

             sampled_data_exact_twin_untreated = design_data_exact_twin_untreated.sample(n=num_samples_needed, replace=True)
             design_data_exact_twin_untreated = pd.concat([design_data_exact_twin_untreated, sampled_data_exact_twin_untreated], ignore_index=True)

             sampled_data_exact_twin_treated = design_data_exact_twin_treated.sample(n=num_samples_needed, replace=True)
             design_data_exact_twin_treated = pd.concat([design_data_exact_twin_treated, sampled_data_exact_twin_treated], ignore_index=True)

        
        data[length] = {'host': design_data_host, 'exact_complementary': design_data_exact_complementary, 'exact_twin': design_data_exact_twin, \
                        'exact_twin_untreated': design_data_exact_twin_untreated, 'exact_twin_treated': design_data_exact_twin_treated}

    return data
