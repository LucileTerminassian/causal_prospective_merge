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
from eig_comp_utils import compute_EIG_causal_closed_form,compute_EIG_obs_closed_form,compute_EIG_causal_from_samples,compute_EIG_obs_from_samples,predictions_in_EIG_causal_form,predictions_in_EIG_obs_form
from xbcausalforest import XBCF
from xbart import XBART


def posterior_mean(X, y,sigma_sq, cov_posterior_inv):
    

    return 1/sigma_sq* cov_posterior_inv @ (X.T @ y)

def posterior_covariance_inv(X, sigma_sq, S0_inv):
    
    
    return (1/sigma_sq * X.T @ X + S0_inv)


class BayesianLinearRegression:

    def __init__(self, prior_hyperparameters, model='linear_reg'):
        self.model = model
        self.prior_hyperparameters = prior_hyperparameters
        self._check_prior_hyperparameters()
        self.sigma_0_sq = self.prior_hyperparameters['sigma_0_sq']
        self.inv_cov = self.prior_hyperparameters['inv_cov_0']
        self.beta = self.prior_hyperparameters['beta_0']
        self.causal_index = None
        if type(self.inv_cov) == torch.Tensor:
            self.cov = torch.inverse(self.inv_cov)
        else:
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
        if beta_0 is None or len(beta_0) != X.shape[1]:
            raise ValueError("beta_0 should be a vector of length d.")


        # Calculate covariance matrix of the posterior distribution
        # old cov_matrix_posterior = np.linalg.inv(sigma_sq_inv_y * np.dot(X.T, X) + sigma_0_sq_inv * np.eye(X.shape[1]))
        inv_cov_matrix_posterior = posterior_covariance_inv(X, self.sigma_0_sq,self.inv_cov)
        if type(inv_cov_matrix_posterior) ==torch.Tensor:
            cov_matrix_posterior = torch.inverse(inv_cov_matrix_posterior)
        else:    
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
        samples = mvn.rvs(n_samples)
        if type(self.inv_cov) == torch.Tensor:
            samples = torch.tensor(samples,dtype=torch.float64)
        return samples
    
    def set_causal_index(self,causal_index):
        self.causal_index = causal_index
        
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
            if type(self.inv_cov) == torch.Tensor:
                if condition_after:
                    conditional_mean =  self.beta[:conditioning_index] + sigma_c @ (torch.inverse(sigma_b)@  (conditioning_vec - self.beta[conditioning_index:]))
                else:
                    conditional_mean = self.beta[conditioning_index:] + sigma_c.T @ (torch.inverse(sigma_a)@  (conditioning_vec - self.beta[:conditioning_index]))

                mvn = multivariate_normal(mean=conditional_mean, cov=conditional_cov)

                return torch.tensor(mvn.rvs(n_samples),dtype=torch.float64)
            else:
                if condition_after:
                    conditional_mean =  self.beta[:conditioning_index] + sigma_c @ (np.linalg.inv(sigma_b)@  (conditioning_vec - self.beta[conditioning_index:]))
                else:
                    conditional_mean = self.beta[conditioning_index:] + sigma_c.T @ (np.linalg.inv(sigma_a)@  (conditioning_vec - self.beta[:conditioning_index]))

                mvn = multivariate_normal(mean=conditional_mean, cov=conditional_cov)

                return mvn.rvs(n_samples)
    
        return conditional_sampling_func
    
    def closed_form_obs_EIG(self,X):
            return compute_EIG_obs_closed_form(X, self.cov, self.sigma_0_sq**(1/2))
    
    def closed_form_causal_EIG(self,X):
        if self.causal_index is None:
            raise ValueError("Must set causal index")
        return compute_EIG_causal_closed_form(X, self.cov, self.sigma_0_sq**(1/2), self.causal_index)
    
    
    def samples_obs_EIG(self,X,n_samples_outer_expectation,n_samples_inner_expectation):
            n_samples = n_samples_outer_expectation*(n_samples_inner_expectation+1)
            posterior_samples = self.posterior_sample(n_samples=n_samples)
            predicitions = posterior_samples @ X.T
            predictions_in_form = predictions_in_EIG_obs_form(predicitions, n_samples_outer_expectation, n_samples_inner_expectation)   
            return compute_EIG_obs_from_samples(predictions_in_form, self.sigma_0_sq**(1/2))
     
    def samples_causal_EIG(self,X,n_samples_outer_expectation,n_samples_inner_expectation):
            sample_func = self.return_conditional_sample_function(self.causal_index)
            posterior_samples = self.posterior_sample(n_samples=n_samples_outer_expectation)
            prediction_func = lambda beta: beta @ (X).T
            predictions_paired = predictions_in_EIG_causal_form(pred_func=prediction_func, theta_samples=posterior_samples, theta_sampling_function=sample_func,n_non_causal_expectation= n_samples_inner_expectation,causal_param_first_index= self.causal_index)
            
            n_samples = n_samples_outer_expectation*(n_samples_inner_expectation+1)
            posterior_samples = self.posterior_sample(n_samples=n_samples)
            predicitions = posterior_samples @ X.T
            predictions_upaired = predictions_in_EIG_obs_form(predicitions, n_samples_outer_expectation, n_samples_inner_expectation) 
            return compute_EIG_causal_from_samples(predictions_upaired,predictions_paired, self.sigma_0_sq**(1/2))
    
class BayesianCausalForest():
    
    def __init__(self, prior_hyperparameters):
        self.sigma_0_sq = prior_hyperparameters['sigma_0_sq']
        self.p_categorical_pr = prior_hyperparameters['p_categorical_pr']
        self.p_categorical_trt= prior_hyperparameters['p_categorical_trt']
        self.propensity_is_fit = False
        self.data_is_stored = False
    
    # def set_model_atrs(self,**kwargs):
    #             for k,v in kwargs.items():
    #                 setattr(self.model, k, v)
    
    def store_train_data(self,X,Y,T):
        self.data_stored =True
        self.X_train = X
        self.Y_train = Y
        self.T_train = T
        self.X_train_prog = X
    
    def fit_propensity_model(self,num_trees=100, num_sweeps=80, burnin=15,**kwargs):
        
        if not self.data_is_stored:
             assert("Must store training data first")

        self.prop_model = XBART(num_trees, num_sweeps, burnin,kwargs)
        self.prop_model.fit(self.X_train,self.T_train)

        T_pred_train = self.prop_model.predict(self.X_train)
        self.X_train_prog = np.concatenate([self.X_train,T_pred_train.reshape(-1,1)],axis=1)

        return None
    
    
    def posterior_sample_predictions(self, X, T , n_samples,return_predictions=True,**kwargs):

        """"Returns n sample predictions from the posterior"""

        if not self.data_is_stored:
             assert("Must store training data first")
        

        self.model = XBCF(num_sweeps=n_samples,p_categorical_pr = self.p_categorical_pr,p_categorical_trt = self.p_categorical_trt,**kwargs)

        self.model.fit(
                x_t=self.X_train, # Covariates treatment effect
                x=self.X_train_prog, # Covariates outcome (including propensity score)
                y=self.Y_train,  # Outcome
                z=self.T_train, # Treatment group
                )
        
            
        if self.propensity_is_fit:
            T_pred = self.prop_model.predict(self.X)
            X1 = np.concatenate([self.X,T_pred.reshape(-1,1)],axis=1)
        
        else:
            X1 = X

        predictions = self.model.predict(X,X1=X, return_mean=False, return_muhat=return_predictions)
        
        if return_predictions:
            predictions_0 = np.expand_dims(predictions[0],axis=2)
            predictions_1 = np.expand_dims(predictions[1],axis=2)
            predictions = np.concatenate([predictions_0,predictions_1],axis=2)
            
            predictions = predictions[np.arange((predictions.shape[0])),:,T]
        
        return predictions
    
    def samples_obs_EIG(self,X,T, n_samples_outer_expectation,n_samples_inner_expectation):
            n_samples = n_samples_outer_expectation*(n_samples_inner_expectation+1)
            predicitions = self.posterior_sample_predictions(X=X, T=T,  n_samples=n_samples)
            predictions_in_form = predictions_in_EIG_obs_form(predicitions.T, n_samples_outer_expectation, n_samples_inner_expectation)   
            return compute_EIG_obs_from_samples(predictions_in_form, self.sigma_0_sq**(1/2))