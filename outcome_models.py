import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde,multivariate_normal
import sys
import torch
torch.set_default_tensor_type(torch.FloatTensor) # set the default to float32
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


def posterior_mean(X, y,sigma_sq, cov_posterior_inv):
    

    return 1/sigma_sq* cov_posterior_inv @ (X.T @ y)

def posterior_covariance_inv(X, sigma_sq, S0_inv):
    
    
    return (1/sigma_sq * np.dot(X.T, X) + S0_inv)


class BayesianLinearRegression:

    def __init__(self, prior_hyperparameters, model='linear_reg'):
        self.model = model
        self.prior_hyperparameters = prior_hyperparameters
        self._check_prior_hyperparameters()
        self.sigma_0_sq = self.prior_hyperparameters['sigma_0_sq']
        self.inv_cov = self.prior_hyperparameters['inv_cov_0']
        self.beta = self.prior_hyperparameters['beta_0']
        self.cov = np.linalg.inv(self.inv_cov)

    def _check_prior_hyperparameters(self):
        
        if not isinstance(self.prior_hyperparameters, dict):
            raise ValueError("Prior hyperparameters should be a dictionary.")
        
        if 'beta_0' not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'beta_0'.")
        
        #This should be a matrix of size pxp denoting the covariance in the prior
        if 'inv_cov_0' not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'inv_cov_0'.")
        
        if 'sigma_0_sq' not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'sigma_0_sq'.")
        
        # Ensure std_squared_0 is a scalar
        if not isinstance(self.prior_hyperparameters['sigma_0_sq'], (int, float)):
            raise ValueError("sigma_0_sq should be a scalar.")

    def fit(self, X, Y):

        n, d = X.shape

        sigma_0_sq = self.prior_hyperparameters['sigma_0_sq']
        inv_cov_0 = self.prior_hyperparameters['inv_cov_0']
        beta_0 = self.prior_hyperparameters['beta_0']
        if beta_0 is None or len(beta_0) != np.shape(X)[1]:
            raise ValueError("beta_0 should be a vector of length d.")


        # Calculate covariance matrix of the posterior distribution
        # old cov_matrix_posterior = np.linalg.inv(sigma_sq_inv_y * np.dot(X.T, X) + sigma_0_sq_inv * np.eye(X.shape[1]))
        inv_cov_matrix_posterior = posterior_covariance_inv(X, self.sigma_0_sq,self.inv_cov)

        cov_matrix_posterior = np.linalg.inv(inv_cov_matrix_posterior)

        # Calculate mean vector of the posterior distribution
        # old beta_posterior = np.dot(np.dot(cov_matrix_posterior, X.T), Y) * sigma_sq_inv_y + np.dot(cov_matrix_posterior, beta_0)
        beta_posterior = posterior_mean(X, Y, self.sigma_0_sq, cov_matrix_posterior)
        
        # Prepare posterior parameters dictionary
        dict_posterior_parameters = {
            'posterior_mean': beta_posterior,  
            'posterior_cov_matrix': cov_matrix_posterior,
        }

        self.beta = beta_posterior
        self.cov = cov_matrix_posterior
        self.inv_cov = inv_cov_matrix_posterior

        return dict_posterior_parameters
    
    def posterior_sample(self, n_samples):

        """"Returns n samples from the posterior"""
        mvn = multivariate_normal(mean=self.beta, cov=self.cov)
        return mvn.rvs(n_samples)
    
    def return_conditional_sample_function(self, conditioning_index,condition_after=True):

        """"Returns a function to sample from the conditional posterior"""
        
        
        sigma_a = self.cov[:conditioning_index, :conditioning_index]
        sigma_b = self.cov[conditioning_index:, conditioning_index:]
        sigma_c = self.cov[:conditioning_index, conditioning_index:]
         
        if condition_after:
            conditional_cov = sigma_a - sigma_c @ sigma_b @ sigma_c.T

        else:
            conditional_cov = sigma_b - sigma_c.T @ sigma_b @ sigma_c
        

        def conditional_sampling_func(conditioning_vec,n_samples):

            if condition_after:
                conditional_mean =  self.beta[:conditioning_index] + sigma_c @ (np.linalg.inv(sigma_b)@  (conditioning_vec - self.beta[conditioning_index:]))
            else:
                conditional_mean = self.beta[conditioning_index:] + sigma_c.T @ (np.linalg.inv(sigma_a)@  (conditioning_vec - self.beta[:conditioning_index]))

            mvn = multivariate_normal(mean=conditional_mean, cov=conditional_cov)

            return mvn.rvs(n_samples)
    
        return conditional_sampling_func
        