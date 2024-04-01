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
    for i in range(dim_x):
        x = x_distributions[i]
        dict_x[f"X{i}"] = x
    X = pd.DataFrame(dict_x)
    
    # Generate T
    T = np.random.randint(0, 2, size=n_global)
    
    return X, T

# Probability functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to generate host and mirror data
def generate_host_and_mirror(X, T, f_assigned_to_host, n_host, n_mirror, power_x, power_x_t, outcome_function, std_true_y):

    n_global, d_global = np.shape(X)[0], np.shape(X)[1]
    # Initialize dictionaries
    data_host = {'X' + str(i): [] for i in range(d_global)}
    data_mirror = {'X' + str(i): [] for i in range(d_global)}

    # Add 'T' key to each dictionary
    data_host['T'] = []
    data_mirror['T'] = []

    if n_host + n_mirror > n_global:
        print('n_host + n_mirror > n_rct')
        return 
    
    for i in range(n_global):
        proba_assigned_to_host = f_assigned_to_host(X.iloc[i]['X0'], X.iloc[i]['X1'], T[i], np.random.normal())
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)
        if is_assigned_to_host == 1:
            if len(data_host['X0']) < n_host :
                for j in range(d_global):
                    data_host['X' + str(j)].append(X.iloc[i]['X' + str(j)])
                data_host['T'].append(T[i])
            else:
                if len(data_mirror['X0']) == n_mirror:
                    break

        else:
            if len(data_mirror['X0']) < n_xmirror :
                for j in range(d_global):
                    data_mirror['X' + str(j)].append(X.iloc[i]['X' + str(j)])
                data_mirror['T'].append(T[i])
            else:
                if len(data_mirror['X0']) == n_mirror:
                    break

    data_host = pd.DataFrame(data_host)
    data_mirror = pd.DataFrame(data_mirror)

    design_data_host = generate_design_matrix(data_host, power_x, power_x_t)
    design_data_mirror = generate_design_matrix(data_mirror, power_x, power_x_t)

    design_data_host = add_outcome(design_data_host, outcome_function, std_true_y)
    design_data_mirror = add_outcome(design_data_mirror, outcome_function, std_true_y)
    
    return design_data_host, design_data_mirror


# def generate_host_and_exact_mirror(X, T, f_assigned_to_host, n_host, n_mirror, power_x, power_x_t, outcome_function, std_true_y):

#     n_global = len(X)
#     data_host = {'X0': [], 'X1': [], 'T': [] }
#     data_mirror = {'X0': [], 'X1': [], 'T': [] }
    
#     for i in range(n_host):
#         proba_assigned_to_host = f_assigned_to_host(X.iloc[i]['X0'], X.iloc[i]['X1'], T[i], np.random.normal())
#         is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)
#         if is_assigned_to_host == 1:
#             if len(data_host['X0']) < n_host :
#                 data_host['X0'].append(X.iloc[i]['X0'])
#                 data_host['X1'].append(X.iloc[i]['X1'])
#                 data_host['T'].append(T[i])

    
#     data_mirror['X0'] = data_host['X0']
#     data_mirror['X1'] = data_host['X1']
#     complementary_treat = [1 if bit == 0 else 0 for bit in data_host['T']]
#     data_mirror['T'] = complementary_treat

#     data_host = pd.DataFrame(data_host)
#     data_mirror = pd.DataFrame(data_mirror)

#     design_data_host = generate_design_matrix(data_host, power_x, power_x_t)
#     design_data_mirror = generate_design_matrix(data_mirror, power_x, power_x_t)

#     design_data_host = add_outcome(design_data_host, outcome_function, std_true_y)
#     design_data_mirror = add_outcome(design_data_mirror, outcome_function, std_true_y)
    
#     return design_data_host, design_data_mirror


# Function to generate host2 data
def generate_cand2(X, T, f_assigned_to_cand2, n_cand2, power_x, power_x_t, outcome_function, std_true_y):
   
    n_global, d_global = np.shape(X)[0], np.shape(X)[1]
    # Initialize dictionaries
    data_cand2 = {'X' + str(i): [] for i in range(d_global)}

    # Add 'T' key to each dictionary
    data_cand2['T'] = []


    if n_cand2 > n_global:
        print('n_cand2 > n_rct')
        return

    for i in range(n_global):
        proba_assigned_to_cand2 = f_assigned_to_cand2(X.iloc[i]['X0'], X.iloc[i]['X1'], T[i], np.random.normal())
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 == 1:
            if len(data_cand2['X0']) < n_cand2 :
                for j in range(d_global):
                    data_cand2['X' + str(j)].append(X.iloc[i]['X' + str(j)])
                data_cand2['T'].append(T[i])
            else:
                break
    
    data_cand2 = pd.DataFrame(data_cand2)

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
    X = data.drop(columns=['T']).values
    T = data['T'].values
    eps = np.random.normal(size=n, scale=scale)

    Y = outcome_function(X, T, eps)
    data['Y'] = Y

    return data