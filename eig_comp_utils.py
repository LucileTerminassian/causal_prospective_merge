import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import scipy.stats._covariance as cov
import torch
from tqdm import tqdm

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


def log_posterior_predictive(
    y, y_pred_theta_samples, covariance, generating_prediction=None
):
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
    n_e = len(y)

    log_likelihood_list = []
    if generating_prediction is not None:
        log_likelihood_list.append(
            multivariate_normal_log_likelihood(y, generating_prediction, covariance)
        )
    for y_pred in y_pred_theta_samples:
        log_likelihood_list.append(
            multivariate_normal_log_likelihood(y, y_pred, covariance)
        )
    return logsumexp(log_likelihood_list) - np.log(len(log_likelihood_list))


def predictions_in_EIG_obs_form(Y_pred_vec, n_outer_expectation, m_inner_expectation):
    """ "Gets samples in the correct form for EIG computation
    Y_pred_vec: predictions from the model over many theta
    n_outer_expectation: number of samples for outer expectation
    m_inner_expectation: number of samples for inner expectation
    """

    if n_outer_expectation * (m_inner_expectation + 1) != len(Y_pred_vec):
        assert "n * m must be the length of the pred vector"
    predictions_list = []

    for i in range(n_outer_expectation):
        predictions_list.append(
            (
                Y_pred_vec[i],
                Y_pred_vec[
                    m_inner_expectation * i
                    + n_outer_expectation : m_inner_expectation * (i + 1)
                    + n_outer_expectation
                ],
            )
        )
    return predictions_list


def predictions_in_EIG_causal_form(
    pred_func,
    theta_samples,
    theta_sampling_function,
    n_non_causal_expectation,
    causal_param_first_index,
):

    paired_predictions = []

    for theta in theta_samples:
        theta_causal = theta[causal_param_first_index:]
        thetas_samples_non_causal = theta_sampling_function(
            theta_causal, n_non_causal_expectation
        )
        if type(theta_causal) == torch.Tensor:
            predictions = [
                pred_func(torch.concatenate([(theta_noncausal), (theta_causal)]))
                for theta_noncausal in thetas_samples_non_causal
            ]
        else:
            predictions = [
                pred_func(
                    np.concatenate([np.array(theta_noncausal), np.array(theta_causal)])
                )
                for theta_noncausal in thetas_samples_non_causal
            ]
        paired_predictions.append((pred_func(theta), predictions))

    return paired_predictions


def calc_posterior_predictive_entropy(pred_list, sigma, lower=False):
    n_e = len(pred_list[0][0])  # old len(pred_list[0][0])
    covariance = cov.CovViaDiagonal(sigma**2 * np.ones(n_e))
    sample_list = []

    for y_pred, y_pred_multiple in tqdm(pred_list):
        mvn = multivariate_normal(mean=y_pred, cov=covariance)
        y_sample = mvn.rvs()
        if lower:
            sample_list.append(
                log_posterior_predictive(y_sample, y_pred_multiple, covariance, y_pred)
            )
        else:
            sample_list.append(
                log_posterior_predictive(y_sample, y_pred_multiple, covariance, None)
            )
    return -(sum(sample_list) / len(sample_list))


def compute_EIG_causal_from_samples(pred_list_unpaired, pred_list_paired, sigma):
    """ " Function to calculate causal information gain"""
    # n_e = len(pred_list_unpaired[0][0])  # old len(pred_list_unpaired[0][0])
    # return calc_posterior_predictive_entropy(
    #     pred_list_unpaired, sigma
    # ) - calc_posterior_predictive_entropy(pred_list_paired, sigma)
    n_e = len(pred_list_unpaired[0][0])  # old len(pred_list[0][0])
    covariance = cov.CovViaDiagonal(sigma**2 * np.ones(n_e))
    sample_list = []

    for (y_pred, y_pred_multiple_paired),(_,y_pred_multiple_unpaired) in tqdm(zip(pred_list_paired,pred_list_unpaired)):
        mvn = multivariate_normal(mean=y_pred, cov=covariance)
        y_sample = mvn.rvs()
        sample_list.append(
            log_posterior_predictive(
                y_sample, y_pred_multiple_paired, covariance, y_pred
            )
            - log_posterior_predictive(
                y_sample, y_pred_multiple_unpaired, covariance, y_pred
            ))
    return sum(sample_list) / len(sample_list)



def compute_EIG_obs_from_samples(pred_list, sigma, lower=False):
    n_e = len(pred_list[0][0])
    return calc_posterior_predictive_entropy(pred_list, sigma, lower) - n_e / 2 * (
        1 + np.log(2 * np.pi * sigma**2)
    )


def compute_EIG_obs_from_samples_alt(pred_list, sigma, lower=False):
    n_e = len(pred_list[0][0])  # old len(pred_list[0][0])
    covariance = cov.CovViaDiagonal(sigma**2 * np.ones(n_e))
    sample_list = []

    for y_pred, y_pred_multiple in pred_list:
        mvn = multivariate_normal(mean=y_pred, cov=covariance)
        y_sample = mvn.rvs()
        if lower:
            sample_list.append(
                multivariate_normal_log_likelihood(y_sample, y_pred, covariance)
                - log_posterior_predictive(
                    y_sample, y_pred_multiple, covariance, y_pred
                )
            )
        else:
            sample_list.append(
                multivariate_normal_log_likelihood(y_sample, y_pred, covariance)
                - log_posterior_predictive(y_sample, y_pred_multiple, covariance, None)
            )
    return sum(sample_list) / len(sample_list)


def compute_EIG_obs_closed_form(X, cov_matrix_prior, sigma_rand):

    n_e = len(X)

    sign, log_det_term = np.linalg.slogdet(
        X @ ((cov_matrix_prior) @ X.T) + (sigma_rand**2) * np.eye(n_e)
    )
    log_sigma_term = n_e * np.log(sigma_rand)
    eig = 0.5 * log_det_term - log_sigma_term

    return eig


def compute_EIG_obs_closed_form_alt(X, cov_matrix_prior, sigma_rand):
    sign, log_det_term_post = np.linalg.slogdet(
        (1 / sigma_rand**2) * X.T @ X + np.linalg.inv(cov_matrix_prior)
    )
    sign, log_det_term_prior = np.linalg.slogdet(np.linalg.inv(cov_matrix_prior))
    return 1 / 2 * log_det_term_post - 1 / 2 * log_det_term_prior


def compute_EIG_causal_closed_form(
    X, cov_matrix_prior, sigma_rand, causal_param_first_index: int
) -> float:

    n_e = len(X)
    num_causal = X.shape[1] - causal_param_first_index
    num_non_causal = causal_param_first_index

    # shapes: [num_nc, num_nc], [num_c, num_c], [num_nc, num_c], respectively
    sigma_a = cov_matrix_prior[:causal_param_first_index, :causal_param_first_index]
    sigma_b = cov_matrix_prior[causal_param_first_index:, causal_param_first_index:]
    sigma_c = cov_matrix_prior[:causal_param_first_index, causal_param_first_index:]
    assert (
        sigma_c.shape == (num_non_causal, num_causal)
        and sigma_a.shape == (num_non_causal, num_non_causal)
        and sigma_b.shape == (num_causal, num_causal)
    ), "Shape mismatch!"

    # I think this is wrong, I think this gives cov_matrix causal?
    # changing name to cov_matrix_prior_c; also commenting it out
    # cov_matrix_prior_c = sigma_b - np.dot(
    #     np.dot(sigma_c.T, np.linalg.inv(sigma_a)), sigma_c
    # )  # [num_causal, num_causal]
    # assert cov_matrix_prior_c.shape == (num_causal, num_causal)

    # this should be the non-causal I think... [?!]
    cov_matrix_prior_nc = sigma_a - np.dot(
        np.dot(sigma_c, np.linalg.inv(sigma_b)), sigma_c.T
    )  # [num_non_causal, num_non_causal]
    assert cov_matrix_prior_nc.shape == (num_non_causal, num_non_causal)

    sign, log_gen_term = np.linalg.slogdet(
        X @ ((cov_matrix_prior) @ X.T) + (sigma_rand**2) * np.eye(n_e)
    )  # scalar
    # phi_nc(X) takes only non-causal columns in X
    phi_nc_X = X[:, :causal_param_first_index]  # [n_e, number of non-causal]

    sign, log_nc_term = np.linalg.slogdet(
        phi_nc_X @ ((cov_matrix_prior_nc) @ phi_nc_X.T) + (sigma_rand**2) * np.eye(n_e)
    )  # scalar

    eig = 0.5 * (log_gen_term - log_nc_term)

    return eig
