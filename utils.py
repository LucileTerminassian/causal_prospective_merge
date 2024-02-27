import numpy as np
from scipy.stats import multivariate_normal

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

def predictive_normal_log_likelihood(y, y_pred_vec, covariance):
    ### THIS IS THE FUNCTION TO BE IMPROVED 
    """
    Compute the log likelihood of the posterior predictive  .

    Parameters:
    - y: y values to be input into likelihood as a numpy array (n), where n is the number of samples.
    - y_pred_vec: Predictions as a numpy array (n x d), where d is the number of predictions to be averaged over. 
    - covariance: Covariance matrix of the multivariate normal distribution.

    Returns:
    - log_predictive_likelihood: Log likelihood of the posterior predictive at y.
    """
    likelihood_list = []
    for y_pred in y_pred_vec.T:
        likelihood_list.append(multivariate_normal_likelihood(y, y_pred, covariance))
    return np.log(sum(likelihood_list)/len(likelihood_list))