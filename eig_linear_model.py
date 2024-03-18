import torch
from torch.distributions import Distribution
import torch.distributions as dist
from torch import Tensor

NUM_COVARIATES = 3
NUM_OBS = 10
COVARIATES = dist.MultivariateNormal(
    torch.zeros(NUM_COVARIATES), torch.eye(NUM_COVARIATES)
).sample(torch.Size([NUM_OBS]))
PRIOR = dist.MultivariateNormal(torch.zeros(NUM_COVARIATES), torch.eye(NUM_COVARIATES))
OBSERVATION_SIGMA = 1.0


def model(
    prior: dist.MultivariateNormal = PRIOR,
    observation_sigma: float = OBSERVATION_SIGMA,
    covariates_X: Tensor = COVARIATES,
) -> tuple[Tensor, Tensor]:
    """SAMPLE FROM THE LINEAR CONJUGATE MODEL:

        y = X @ params + eps, where eps ~ N(0, observation_sigma^2)

    Args:
        prior: Multivariate normal. Defaults to PRIOR which is standard Gsn.
        observation_sigma: Observation noise. Defaults to OBSERVATION_SIGMA=1.0
        covariates_X: Covariates (aka design matrix). Defaults to COVARIATES, sampled
            from multivariate GSN

    Returns:
        Tuple of outcomes, and sampled parameters
    """
    params = prior.sample()
    assert params.shape == covariates_X.shape[1:], "Shape mismatch!"
    y = dist.Normal(covariates_X @ params, observation_sigma).sample()
    return y, params


def compute_posterior(
    observations_y: Tensor,
    prior: dist.MultivariateNormal = PRIOR,
    covariates_X: Tensor = COVARIATES,
    observation_sigma: float = OBSERVATION_SIGMA,
) -> dist.MultivariateNormal:
    """
    Compute the posterior distribution on the parameters given the observations.
    """
    precision_matrix = prior.precision_matrix
    XT_X = covariates_X.T @ covariates_X
    XT_Y = covariates_X.T @ observations_y

    new_covmat = torch.inverse(precision_matrix + XT_X / observation_sigma**2)
    new_loc = new_covmat @ (XT_Y / observation_sigma**2)
    return dist.MultivariateNormal(new_loc, new_covmat)


def eig_closed_form(
    prior: dist.MultivariateNormal = PRIOR,
    covariates_X: Tensor = COVARIATES,
    observation_sigma: float = OBSERVATION_SIGMA,
    n_samples: int = 32,
) -> Tensor:
    ## can vectorize over n_samples, but let's loop to not have to deal with shapes now
    temp = torch.empty(n_samples)

    for i in range(n_samples):
        # get outcomes
        obs_y, params = model(prior, observation_sigma, covariates_X)

        posterior = compute_posterior(
            observations_y=obs_y,
            prior=prior,
            covariates_X=covariates_X,
            observation_sigma=observation_sigma,
        )
        temp[i] = posterior.entropy()

    return prior.entropy() - temp.mean()


def eig_from_samples_PCE(
    prior: dist.MultivariateNormal = PRIOR,
    covariates_X: Tensor = COVARIATES,
    observation_sigma: float = OBSERVATION_SIGMA,
    n_outer_samples: int = 4096,
    n_inner_samples: int = 128,
    is_lower: bool = True,
) -> Tensor:

    temp = torch.empty(n_outer_samples)  # hold the evaluations of the density ratios

    for i in range(n_outer_samples):
        # run model to get params and outcomes
        obs_y, params = model(prior, observation_sigma, covariates_X)

        # this is the "primary" model
        primary_loc = params @ covariates_X.T  # [n_obs]
        likelihood_model = dist.Normal(loc=primary_loc, scale=observation_sigma)
        numerator_logprob = likelihood_model.log_prob(obs_y).sum()  # [n_obs] -> [1]

        # now resample params for the evaluation of the denominator:
        resampled_params = prior.sample(torch.Size([n_inner_samples]))
        # need to compute logprobs under these new params
        reampled_loc = resampled_params @ covariates_X.T  # [n_inner_samples, n_obs]
        denominator_logprob = (
            dist.Normal(loc=reampled_loc, scale=observation_sigma)
            .log_prob(obs_y.unsqueeze(0))
            .sum(dim=1)  # [n_inner_samples, n_obs] -> [n_inner_samples]
        )  # [n_inner_samples]

        if is_lower:
            # append the logprob from the numerator to the denominator
            # (so we have n_inner_samples + 1)
            denominator_logprob = torch.cat(
                [denominator_logprob, numerator_logprob.unsqueeze(0)]
            )
            const = torch.log(torch.tensor(n_inner_samples + 1.0))
        else:
            const = torch.log(torch.tensor(1.0 * n_inner_samples))
        denominator_logprob = torch.logsumexp(denominator_logprob, dim=0)
        temp[i] = numerator_logprob - denominator_logprob

    return temp.mean() + const


if __name__ == "__main__":
    y, params = model(
        PRIOR, observation_sigma=OBSERVATION_SIGMA, covariates_X=COVARIATES
    )
    print("COVARIATES.shape", COVARIATES.shape)
    print("params.shape", params.shape)
    print("y.shape", y.shape)

    posterior = compute_posterior(y, PRIOR, COVARIATES, OBSERVATION_SIGMA)
    print("Closed form Posterior mean =", posterior.mean)
    print("Actual parameter values    = ", params)

    print("Closed form Posterior covariance_matrix", posterior.covariance_matrix)
    print("\n")

    eig = eig_closed_form(PRIOR, COVARIATES, OBSERVATION_SIGMA)
    print("EIG closed form", eig)

    eig_samples_lower = eig_from_samples_PCE(
        PRIOR, COVARIATES, OBSERVATION_SIGMA, 32, 1024, True
    )
    eig_samples_upper = eig_from_samples_PCE(
        PRIOR, COVARIATES, OBSERVATION_SIGMA, 32, 1024, False
    )
    print("EIG from samples, lower", eig_samples_lower)
    print("EIG from samples, upper", eig_samples_upper)
