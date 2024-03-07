import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def multivariate_normal_likelihood(data, mean, covariance):
    """
    Compute the likelihood of multivariate normal distribution.

    Parameters:
    - data: Data points as a numpy array (n x d), where n is the number of samples and d is the dimensionality.
    - mean: Mean vector of the multivariate normal distribution.
    - covariance: Covariance matrix of the multivariate normal distribution.

    Returns:
    - likelihood: Likelihood of the data given the multivariate normal distribution.
    """
    mvn = multivariate_normal(mean=mean, cov=covariance)
    likelihood = mvn.pdf(data)
    return likelihood


def multivariate_normal_log_likelihood(data, mean, covariance):
    """
    Compute the log likelihood of multivariate normal distribution.

    Parameters:
    - data: Data points as a numpy array (n x d), where n is the number of samples and d is the dimensionality.
    - mean: Mean vector of the multivariate normal distribution.
    - covariance: Covariance matrix of the multivariate normal distribution.

    Returns:
    - log_likelihood: Log likelihood of the data given the multivariate normal distribution.
    """
    mvn = multivariate_normal(mean=mean, cov=covariance)
    log_likelihood = mvn.logpdf(data)
    return log_likelihood

def log_posterior_predictive(y,y_pred_theta_samples, covariance):
    
    """
    Compute the log likelihood of the posterior predictive  .

    Parameters:
    - y: y values to be input into likelihood as a numpy array (n), where n is the number of samples.
    - y_pred_theta_samples: f_{theta}(X,t) for samples theta from posterior. 
        Should be a numpy array (n x d), where d is the number of posterior samples averaged over. 
    - covariance: Covariance matrix of the multivariate normal distribution.

    Returns:
    - log posterior predictive likelihood: A single point estimate of log(P(y|X,t)).
    """
    log_likelihood_list = []
    for y_pred in y_pred_theta_samples:
        log_likelihood_list.append(multivariate_normal_log_likelihood(y, y_pred, covariance))
    return logsumexp(log_likelihood_list) - np.log(len(log_likelihood_list))

def samples_in_EIG_form (Y_pred_vec, n_outer_expectation, m_inner_expectation):
    
    """"Gets samples in the correct form for EIG computation
    Y_pred_vec: predictions from the model over many theta
    n_outer_expectation: number of samples for outer expectation
    m_inner_expectation: number of samples for inner expectation
    """

    if n_outer_expectation * m_inner_expectation != len(Y_pred_vec):
        assert("n * m must be the length of the pred vector")
    predictions_list = []

    for i in range(n_outer_expectation):
        predictions_list.append((Y_pred_vec[i], \
             Y_pred_vec[ m_inner_expectation * i + n_outer_expectation: m_inner_expectation * (i+1) + n_outer_expectation]))
    return predictions_list

def resampling_theta_non_causal (theta_samples, theta_sampling_function, n_non_causal_expectation, causal_param_first_index):

    resampled_thetas = []

    for theta in theta_samples:
        theta_causal = theta[causal_param_first_index:]
        thetas_samples_non_causal = theta_sampling_function(theta_causal, n_non_causal_expectation)
        resampled_thetas.append((theta, thetas_samples_non_causal))
        
    return resampled_thetas

# def linear_theta_sampling(theta_causal, n_non_causal_expectation):

    # theta_causal is one set of causal parameters
    # thetas_samples_non_causal = []

    #for i in range (n_non_causal_expectation):
        # sample n_non_causal_expectation conditional on theta_causal
        # append this to thetas_samples_non_causal
    # return tuple (theta_causal, thetas_samples_non_causal)

# JAKE write marginalization wrapping function ie computation of EIG_causal




def compute_EIG_obs_from_samples(pred_list, sigma):
    n_e = len(pred_list[0][0])
    covariance = sigma*np.eye((n_e))
    sample_list = []
    
    for y_pred,y_pred_multiple in pred_list:
        mvn = multivariate_normal(mean=y_pred, cov=covariance)
        y_sample = mvn.rvs()
        sample_list.append(log_posterior_predictive(y_sample,y_pred_multiple,covariance))

    return -(sum(sample_list)/len(sample_list)) - n_e/2 * (1 + np.log(2 * np.pi * sigma **2))


def compute_EIG_obs_closed_form(X, cov_matrix_prior, sigma_rand):

    n_e = len(X)
    det_term = np.linalg.det( X @ (np.linalg.inv(cov_matrix_prior) @ X.T) + (sigma_rand**2) * np.eye(n_e)) 
    log_det_term = np.log(det_term)
    log_sigma_term = n_e * np.log(sigma_rand)
    eig = 0.5 * log_det_term - log_sigma_term

    return eig

def compute_EIG_causal_closed_form(X, cov_matrix_prior, sigma_rand, causal_param_first_index):

    n_e = len(X)
    inv_cov_matrix_prior = np.linalg.inv(cov_matrix_prior)
    sigma_a = inv_cov_matrix_prior[:causal_param_first_index, 0:causal_param_first_index]
    sigma_b = inv_cov_matrix_prior[causal_param_first_index:, causal_param_first_index:]
    sigma_c = inv_cov_matrix_prior[:causal_param_first_index, causal_param_first_index:]

    cov_matrix_prior_nc = sigma_b - np.dot(np.dot(sigma_c.T, np.linalg.inv(sigma_a) ) , sigma_c)

    gen_term = np.linalg.det( X @ (np.linalg.inv(cov_matrix_prior) @ X.T) + (sigma_rand**2) * np.eye(n_e)) 
    log_gen_term = np.log(gen_term)

    # phi_nc(X) takes only non-causal columns in X
    phi_nc_X = X[:,:causal_param_first_index]
    nc_term = np.linalg.det(phi_nc_X @ (cov_matrix_prior_nc @ phi_nc_X.T) + (sigma_rand**2) * np.eye(n_e))
    log_nc_term = np.log(nc_term)
    eig = 0.5 * (log_gen_term - log_nc_term)

    return eig