import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
import sys
import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


def posterior_mean(X, y, sigma_sq, S0_inv, m0):
    
    A = 1/sigma_sq * np.dot(X.T, X) + S0_inv
    A_inv = np.linalg.inv(A)
    b = 1/sigma_sq * np.dot(X.T, y) + np.dot(S0_inv, m0)
    
    return np.dot(A_inv, b)

def posterior_covariance(X, sigma_sq, S0_inv):
    
    
    return np.linalg.inv(1/sigma_sq * np.dot(X.T, X) + S0_inv)


class BayesianLinearRegression:

    def __init__(self, prior_hyperparameters, model='linear_reg'):
        self.model = model
        self.prior_hyperparameters = prior_hyperparameters
        self._check_prior_hyperparameters()

    def _check_prior_hyperparameters(self):
        
        if not isinstance(self.prior_hyperparameters, dict):
            raise ValueError("Prior hyperparameters should be a dictionary.")
        
        if 'beta_0' not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'beta_0'.")
        
        if 'sigma_0_sq' not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'sigma_0_sq'.")
        
        # Ensure std_squared_0 is a scalar
        if not isinstance(self.prior_hyperparameters['sigma_0_sq'], (int, float)):
            raise ValueError("sigma_0_sq should be a scalar.")

    def fit(self, X, Y):

        n, d = X.shape

        sigma_0_sq = self.prior_hyperparameters['sigma_0_sq']
        beta_0 = self.prior_hyperparameters['beta_0']
        if beta_0 is None or len(beta_0) != np.shape(X)[1]:
            raise ValueError("beta_0 should be a vector of length d.")

        sigma_sq_y = np.var(Y)
        # old sigma_sq_inv_y = 1 / sigma_sq_y
        sigma_0_sq_inv = 1 / sigma_0_sq

        # Calculate covariance matrix of the posterior distribution
        # old cov_matrix_posterior = np.linalg.inv(sigma_sq_inv_y * np.dot(X.T, X) + sigma_0_sq_inv * np.eye(X.shape[1]))
        cov_matrix_posterior = posterior_covariance(X, sigma_sq_y, sigma_0_sq_inv)

        # Calculate mean vector of the posterior distribution
        # old beta_posterior = np.dot(np.dot(cov_matrix_posterior, X.T), Y) * sigma_sq_inv_y + np.dot(cov_matrix_posterior, beta_0)
        beta_posterior = posterior_mean(X, Y, sigma_sq_y, sigma_0_sq_inv, beta_0)

        if sigma_sq_y == 0:
            raise ValueError("Variance of Y is zero. Cannot divide by zero.")
        
        # Prepare posterior parameters dictionary
        dict_posterior_parameters = {
            'posterior_mean': beta_posterior,  
            'posterior_cov_matrix': cov_matrix_posterior,
        }

        return dict_posterior_parameters
    