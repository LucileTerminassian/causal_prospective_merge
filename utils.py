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

def samples_in_EIG_form(Y_pred_vec,n_samples_for_expectation,m_samples_for_expectation):
    """"Gets samples in the correct form for EIG computation
    Y_pred_vec: predictions from the model over many theta
    n_samples_for_expectation: number of samples for outer expectation
    m_samples_for_expectation: number of samples for inner expectation
    """

    if n_samples_for_expectation*m_samples_for_expectation != len(Y_pred_vec):
        assert("n*m must be the length of the pred vector")
    predictions_list = []

    for i in range(n_samples_for_expectation):
        predictions_list.append((Y_pred_vec[i],Y_pred_vec[m_samples_for_expectation* i+n_samples_for_expectation: m_samples_for_expectation* (i+1)+n_samples_for_expectation]))
    return predictions_list

def calc_EIG_observational(pred_list,sigma):
    n_e = len(pred_list[0][0])
    covariance = sigma*np.eye((n_e))
    sample_list = []
    for y_pred,y_pred_multiple in pred_list:
        mvn = multivariate_normal(mean=y_pred, cov=covariance)
        y_sample = mvn.rvs()
        sample_list.append(log_posterior_predictive(y_sample,y_pred_multiple,covariance))
    return -(sum(sample_list)/len(sample_list)) - n_e * np.log(sigma)