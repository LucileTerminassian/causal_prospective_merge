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
def generate_host_and_mirror(X, T, f_assigned_to_host):
    n_global = len(X)
    data_host = {'X0': [], 'X1': [], 'T': [] }
    data_mirror = {'X0': [], 'X1': [], 'T': [] }
    
    for i in range(n_global):
        proba_assigned_to_host = f_assigned_to_host(X.iloc[i]['X0'], X.iloc[i]['X1'], T[i], np.random.normal())
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)
        if is_assigned_to_host == 1:
            data_host['X0'].append(X.iloc[i]['X0'])
            data_host['X1'].append(X.iloc[i]['X1'])
            data_host['T'].append(T[i])
        else:
            data_mirror['X0'].append(X.iloc[i]['X0'])
            data_mirror['X1'].append(X.iloc[i]['X1'])
            data_mirror['T'].append(T[i])
    
    data_host = pd.DataFrame(data_host)
    data_mirror = pd.DataFrame(data_mirror)
    
    return data_host, data_mirror

# Function to generate host2 data
def generate_host2(X, T, f_assigned_to_cand2, n_cand2):
    n_global = len(X)
    data_cand2 = {'X0': [], 'X1': [], 'T': []}
    
    for i in range(n_cand2):
        proba_assigned_to_cand2 = f_assigned_to_cand2(X.iloc[i]['X0'], X.iloc[i]['X1'], T[i], np.random.normal())
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 == 1:
            data_cand2['X0'].append(X.iloc[i]['X0'])
            data_cand2['X1'].append(X.iloc[i]['X1'])
            data_cand2['T'].append(T[i])
    
    data_cand2 = pd.DataFrame(data_cand2)
    
    return data_cand2

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