import numpy as np
from scipy.stats import multivariate_normal
import torch


torch.set_default_tensor_type(torch.FloatTensor)  # set the default to float32


from eig_comp_utils import (
    compute_EIG_causal_closed_form,
    compute_EIG_obs_closed_form,
    compute_EIG_causal_from_samples,
    compute_EIG_obs_from_samples,
    predictions_in_EIG_causal_form,
    predictions_in_EIG_obs_form,
    calc_posterior_predictive_entropy,
)
from xbcausalforest import XBCF
from xbart import XBART
from tqdm import tqdm
from cmgp import CMGP


def posterior_mean(X, y, sigma_sq, cov_posterior_inv):
    return 1 / sigma_sq * cov_posterior_inv @ ((X.T) @ y.double())


def posterior_covariance_inv(X, sigma_sq, S0_inv):
    return 1 / sigma_sq * X.T @ X + S0_inv


class BayesianLinearRegression:
    def __init__(self, prior_hyperparameters, model="linear_reg"):
        self.model = model
        self.prior_hyperparameters = prior_hyperparameters
        self._check_prior_hyperparameters()
        self.sigma_0_sq = self.prior_hyperparameters["sigma_0_sq"]
        self.inv_cov = self.prior_hyperparameters["inv_cov_0"]
        self.beta = self.prior_hyperparameters["beta_0"]
        self.causal_index = None
        if type(self.inv_cov) == torch.Tensor:
            self.cov = torch.inverse(self.inv_cov)
        else:
            self.cov = np.linalg.inv(self.inv_cov)
        


    def _check_prior_hyperparameters(self):

        if not isinstance(self.prior_hyperparameters, dict):
            raise ValueError("Prior hyperparameters should be a dictionary.")

        if "beta_0" not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'beta_0'.")

        # This should be a matrix of size pxp denoting the covariance in the prior
        if "inv_cov_0" not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'inv_cov_0'.")

        if "sigma_0_sq" not in self.prior_hyperparameters:
            raise ValueError("Prior hyperparameters should contain key 'sigma_0_sq'.")

        # Ensure std_squared_0 is a scalar
        if not isinstance(self.prior_hyperparameters["sigma_0_sq"], (int, float)):
            raise ValueError("sigma_0_sq should be a scalar.")

    def fit(self, X, Y):

        n, d = X.shape

        sigma_0_sq = self.prior_hyperparameters["sigma_0_sq"]
        inv_cov_0 = self.prior_hyperparameters["inv_cov_0"]
        beta_0 = self.prior_hyperparameters["beta_0"]
        if beta_0 is None or len(beta_0) != X.shape[1]:
            raise ValueError("beta_0 should be a vector of length d.")

        # Calculate covariance matrix of the posterior distribution
        # old cov_matrix_posterior = np.linalg.inv(sigma_sq_inv_y * np.dot(X.T, X) + sigma_0_sq_inv * np.eye(X.shape[1]))
        inv_cov_matrix_posterior = posterior_covariance_inv(
            X, self.sigma_0_sq, self.inv_cov
        )
        if type(inv_cov_matrix_posterior) == torch.Tensor:
            cov_matrix_posterior = torch.inverse(inv_cov_matrix_posterior)
        else:
            cov_matrix_posterior = np.linalg.inv(inv_cov_matrix_posterior)

        # Calculate mean vector of the posterior distribution
        # old beta_posterior = np.dot(np.dot(cov_matrix_posterior, X.T), Y) * sigma_sq_inv_y + np.dot(cov_matrix_posterior, beta_0)
        beta_posterior = posterior_mean(X, Y, self.sigma_0_sq, cov_matrix_posterior)

        # Prepare posterior parameters dictionary
        dict_posterior_parameters = {
            "posterior_mean": beta_posterior,
            "posterior_cov_matrix": cov_matrix_posterior,
        }

        self.beta = beta_posterior
        self.cov = cov_matrix_posterior
        self.inv_cov = inv_cov_matrix_posterior

        return dict_posterior_parameters

    def posterior_sample(self, n_samples):
        """ "Returns n samples from the posterior"""
        mvn = multivariate_normal(mean=self.beta, cov=self.cov)
        samples = mvn.rvs(n_samples, random_state=0)
        if type(self.inv_cov) == torch.Tensor:
            samples = torch.tensor(samples, dtype=torch.float64)
        return samples

    def set_causal_index(self, causal_index):
        self.causal_index = causal_index

    def return_conditional_sample_function(
        self, conditioning_index, condition_after=True
    ):
        """ "Returns a function to sample from the conditional posterior"""

        sigma_a = self.cov[:conditioning_index, :conditioning_index]
        sigma_b = self.cov[conditioning_index:, conditioning_index:]
        sigma_c = self.cov[:conditioning_index, conditioning_index:]

        if condition_after:
            conditional_cov = sigma_a - sigma_c @ sigma_b @ sigma_c.T

        else:
            conditional_cov = sigma_b - sigma_c.T @ sigma_b @ sigma_c

        def conditional_sampling_func(conditioning_vec, n_samples):
            if type(self.inv_cov) == torch.Tensor:
                if condition_after:
                    conditional_mean = self.beta[:conditioning_index] + sigma_c @ (
                        torch.inverse(sigma_b)
                        @ (conditioning_vec - self.beta[conditioning_index:])
                    )
                else:
                    conditional_mean = self.beta[conditioning_index:] + sigma_c.T @ (
                        torch.inverse(sigma_a)
                        @ (conditioning_vec - self.beta[:conditioning_index])
                    )

                mvn = multivariate_normal(mean=conditional_mean, cov=conditional_cov)

                return torch.tensor(
                    mvn.rvs(n_samples, random_state=0), dtype=torch.float64
                )
            else:
                if condition_after:
                    conditional_mean = self.beta[:conditioning_index] + sigma_c @ (
                        np.linalg.inv(sigma_b)
                        @ (conditioning_vec - self.beta[conditioning_index:])
                    )
                else:
                    conditional_mean = self.beta[conditioning_index:] + sigma_c.T @ (
                        np.linalg.inv(sigma_a)
                        @ (conditioning_vec - self.beta[:conditioning_index])
                    )

                mvn = multivariate_normal(mean=conditional_mean, cov=conditional_cov)

                return mvn.rvs(n_samples, random_state=0)

        return conditional_sampling_func

    def closed_form_obs_EIG(self, X):
        return compute_EIG_obs_closed_form(X, self.cov, self.sigma_0_sq ** (1 / 2))

    def closed_form_causal_EIG(self, X):
        if self.causal_index is None:
            raise ValueError("Must set causal index")
        return compute_EIG_causal_closed_form(
            X, self.cov, self.sigma_0_sq ** (1 / 2), self.causal_index
        )

    def samples_obs_EIG(
        self, X, n_samples_outer_expectation, n_samples_inner_expectation
    ):
        if not hasattr(self,"posterior_samples"): 
            n_samples = n_samples_outer_expectation * (n_samples_inner_expectation + 1)
            self.posterior_samples = self.posterior_sample(n_samples=n_samples)
            print("sampling done")
        
        predictions = self.posterior_samples @ X.T
        print("predicted")
        predictions_in_form = predictions_in_EIG_obs_form(
            predictions, n_samples_outer_expectation, n_samples_inner_expectation
        )
        print("predictions in form")
        return compute_EIG_obs_from_samples(
            predictions_in_form, self.sigma_0_sq ** (1 / 2)
        )

    def samples_causal_EIG(
        self, X, n_samples_outer_expectation, n_samples_inner_expectation
    ):
        
        sample_func = self.return_conditional_sample_function(self.causal_index)
        
        if not hasattr(self,"posterior_samples"):
            posterior_samples = self.posterior_sample(n_samples=n_samples_outer_expectation)
        else: 
            posterior_samples = self.posterior_samples[:n_samples_outer_expectation]

        prediction_func = lambda beta: beta @ (X).T
        
        predictions_paired = predictions_in_EIG_causal_form(
            pred_func=prediction_func,
            theta_samples=posterior_samples,
            theta_sampling_function=sample_func,
            n_non_causal_expectation=n_samples_inner_expectation,
            causal_param_first_index=self.causal_index,
        )
        print("conditional_sample_done")
        n_samples = n_samples_outer_expectation * (n_samples_inner_expectation + 1)
        posterior_samples = self.posterior_sample(n_samples=n_samples)
        predicitions = posterior_samples @ X.T
        predictions_upaired = predictions_in_EIG_obs_form(
            predicitions, n_samples_outer_expectation, n_samples_inner_expectation
        )
        return compute_EIG_causal_from_samples(
            predictions_upaired, predictions_paired, self.sigma_0_sq ** (1 / 2)
        )


class BayesianCausalForest:
    def __init__(
        self,
        prior_hyperparameters,
        predictive_model_parameters={},
        conditional_model_param={},
    ):
        self.sigma_0_sq = prior_hyperparameters["sigma_0_sq"]
        self.p_categorical_pr = prior_hyperparameters["p_categorical_pr"]
        self.p_categorical_trt = prior_hyperparameters["p_categorical_trt"]
        self.pred_model_param = predictive_model_parameters
        self.cond_model_param = conditional_model_param
        self.propensity_is_fit = False
        self.data_is_stored = False

    # def set_model_atrs(self,**kwargs):
    #             for k,v in kwargs.items():
    #                 setattr(self.model, k, v)

    def store_train_data(self, X, Y, T):
        self.data_stored = True
        self.X_train = X
        self.Y_train = Y
        self.T_train = T
        self.X_train_prog = X

    def fit_propensity_model(self, num_trees=100, num_sweeps=80, burnin=15, **kwargs):

        if not self.data_is_stored:
            assert "Must store training data first"

        self.prop_model = XBART(num_trees, num_sweeps, burnin, kwargs)
        self.prop_model.fit(self.X_train, self.T_train)
        self.propensity_is_fit = True

        T_pred_train = self.prop_model.predict(self.X_train)
        self.X_train_prog = np.concatenate(
            [self.X_train, T_pred_train.reshape(-1, 1)], axis=1
        )

        return None

    def posterior_sample_predictions(self, X, T, n_samples, return_tau=False):
        """ "Returns n sample predictions from the posterior"""

        if not self.data_is_stored:
            assert "Must store training data first"

        self.model = XBCF(
            num_sweeps=n_samples,
            p_categorical_pr=self.p_categorical_pr,
            p_categorical_trt=self.p_categorical_trt,
            **self.pred_model_param
        )

        self.model.fit(
            x_t=self.X_train,  # Covariates treatment effect
            x=self.X_train_prog,  # Covariates outcome (including propensity score)
            y=self.Y_train,  # Outcome
            z=self.T_train,  # Treatment group
        )

        if self.propensity_is_fit:
            T_pred = self.prop_model.predict(self.X)
            X1 = np.concatenate([self.X, T_pred.reshape(-1, 1)], axis=1)

        else:
            X1 = X

        predictions = self.model.predict(X, X1=X1, return_mean=False, return_muhat=True)

        b = self.model.b
        b_adj = b / (np.expand_dims(b[:, 1] - b[:, 0], axis=1))

        tau_adj = predictions[0] * (b_adj.T[T])

        if return_tau:
            return (tau_adj + predictions[1]).T, predictions[0].T
        else:
            return (tau_adj + predictions[1]).T

    def posterior_conditional_predictions(
        self, X, T, Y_residuals, n_samples, return_tau=False
    ):
        """ "Returns n sample predictions from the posterior"""

        if not self.data_is_stored:
            assert "Must store training data first"

        self.model = XBCF(
            num_trees_trt=0,
            num_sweeps=n_samples,
            p_categorical_pr=self.p_categorical_pr,
            p_categorical_trt=self.p_categorical_trt,
            **self.cond_model_param
        )

        self.model.fit(
            x_t=np.zeros_like(self.X_train),  # Covariates treatment effect
            x=self.X_train_prog,  # Covariates outcome (including propensity score)
            y=Y_residuals,  # Outcome
            z=self.T_train,  # Treatment group
        )

        if self.propensity_is_fit:
            T_pred = self.prop_model.predict(self.X)
            X1 = np.concatenate([self.X, T_pred.reshape(-1, 1)], axis=1)

        else:
            X1 = X

        predictions = self.model.predict(X, X1=X, return_mean=False, return_muhat=True)

        b = self.model.b
        b_adj = b / (np.expand_dims(b[:, 1] - b[:, 0], axis=1))

        tau_adj = predictions[0] * (b_adj.T[T])

        if return_tau:
            return (tau_adj + predictions[1]).T, tau_adj.T
        else:
            return (tau_adj + predictions[1]).T

    def samples_obs_EIG(
        self, X, T, n_samples_outer_expectation, n_samples_inner_expectation
    ):
        n_samples = n_samples_outer_expectation * (n_samples_inner_expectation + 1)
        predicitions = self.posterior_sample_predictions(
            X=X, T=T, n_samples=n_samples, return_tau=False
        )
        predictions_in_form = predictions_in_EIG_obs_form(
            predicitions, n_samples_outer_expectation, n_samples_inner_expectation
        )
        return compute_EIG_obs_from_samples(
            predictions_in_form, self.sigma_0_sq ** (1 / 2)
        )

    def samples_causal_EIG(
        self, X, T, n_samples_outer_expectation, n_samples_inner_expectation
    ):
        n_e = len(T)
        print("Calculating posterior predictive entropy")

        posterior_predictive_entropy = self.samples_obs_EIG(
            X, T, n_samples_outer_expectation, n_samples_inner_expectation
        ) + n_e / 2 * (1 + np.log(2 * np.pi * self.sigma_0_sq))
        predictions, tau_pred = self.posterior_sample_predictions(
            X=X, T=T, n_samples=n_samples_outer_expectation, return_tau=True
        )

        tau_train = self.model.tauhats_adjusted.T

        print("Getting conditional samples")
        causal_sample = []
        for i in tqdm(range(len(tau_train))):
            Y_resid = self.Y_train - tau_train[i]
            conditional_predictions = self.posterior_conditional_predictions(
                X=X, T=T, Y_residuals=Y_resid, n_samples=n_samples_inner_expectation
            )
            conditional_predictions = conditional_predictions
            causal_sample.append(
                (predictions[i] - tau_pred[i], conditional_predictions)
            )
        return posterior_predictive_entropy - calc_posterior_predictive_entropy(
            causal_sample, self.sigma_0_sq ** (1 / 2)
        )

    def joint_EIG_calc(self, X, T, sampling_parameters):

        n_samples_outer_expectation_obs, n_samples_inner_expectation_obs = (
            sampling_parameters["n_samples_outer_expectation_obs"],
            sampling_parameters["n_samples_inner_expectation_obs"],
        )
        n_samples_outer_expectation_caus, n_samples_inner_expectation_caus = (
            sampling_parameters["n_samples_outer_expectation_caus"],
            sampling_parameters["n_samples_inner_expectation_caus"],
        )

        n_samples = n_samples_outer_expectation_obs * (
            n_samples_inner_expectation_obs + 1
        )
        print("Sampling from Posterior")

        predicitions_obs, tau_pred = self.posterior_sample_predictions(
            X=X, T=T, n_samples=n_samples, return_tau=True
        )

        predictions_in_form = predictions_in_EIG_obs_form(
            predicitions_obs,
            n_samples_outer_expectation_obs,
            n_samples_inner_expectation_obs,
        )

        random_sample = np.random.choice(
            np.arange(n_samples), n_samples_outer_expectation_caus
        )
        predictions_sampled, tau_sampled = (
            predicitions_obs[random_sample],
            tau_pred[random_sample],
        )

        tau_train = self.model.tauhats_adjusted.T[random_sample]

        print("Getting conditional samples")
        causal_sample = []
        for i in tqdm(range(len(tau_train))):
            Y_resid = self.Y_train - tau_train[i]
            conditional_predictions = self.posterior_conditional_predictions(
                X=X,
                T=T,
                Y_residuals=Y_resid,
                n_samples=n_samples_inner_expectation_caus,
            )
            conditional_predictions = conditional_predictions + tau_sampled[i]
            causal_sample.append((predictions_sampled[i], conditional_predictions))

        posterior_predictive_entropy = calc_posterior_predictive_entropy(
            predictions_in_form, self.sigma_0_sq ** (1 / 2)
        )

        n_e = len(T)

        results_dict = {
            "Posterior Predictive Entropy": posterior_predictive_entropy,
            "Obs EIG": posterior_predictive_entropy
            - n_e / 2 * (1 + np.log(2 * np.pi * self.sigma_0_sq)),
            "Causal EIG": posterior_predictive_entropy
            - calc_posterior_predictive_entropy(
                causal_sample, self.sigma_0_sq ** (1 / 2)
            ),
        }
        return results_dict

class CausalGP:

    def __init__(self,max_gp_iterations=100,min_var=0.01) -> None:
        self.max_gp_iterations = max_gp_iterations
        self.model = None
        self.min_var = min_var

    def fit(self,X_train,T_train,Y_train):

        self.model = CMGP(X_train, T_train, Y_train, max_gp_iterations = self.max_gp_iterations)

        self.X0_train =  np.array(
                np.hstack([X_train, np.zeros_like(X_train[:, 1].reshape((len(X_train[:, 1]), 1)))])
            )
        self.X1_train =  np.array(
                np.hstack([X_train, np.ones_like(X_train[:, 1].reshape((len(X_train[:, 1]), 1)))])
            )
        return None

    def pred_CATE(self,X_test):
        return self.model.predict(X_test)
    
    def obs_EIG_closed_form(self,X,T):
        
        X0 = X[T==0]
        X0 = np.array(
                np.hstack([X0, np.zeros_like(X0[:, 1].reshape((len(X0[:, 1]), 1)))])
            )
        X1 = X[T==1]
        X1 = np.array(
    
    np.hstack([X1, np.ones_like(X1[:, 1].reshape((len(X1[:, 1]), 1)))])
            )
        
        X0_shape = X0.shape
        X1_shape = X1.shape
        noise_dict_0 = {
            "output_index": X0[:, X0_shape[1] - 1]
            .reshape((X0_shape[0], 1))
            .astype(int)}
        noise_dict_1 = {
            "output_index": X1[:, X1_shape[1] - 1]
            .reshape((X1_shape[0], 1))
            .astype(int)}
        
        Sigma_1 = np.block(
            [[self.model.model.posterior_covariance_between_points(X0,X0,Y_metadata=noise_dict_0), self.model.model.posterior_covariance_between_points(X0,X1,include_likelihood=False) ],
            [self.model.model.posterior_covariance_between_points(X1,X0,include_likelihood=False), self.model.model.posterior_covariance_between_points(X1,X1,Y_metadata=noise_dict_1) ]]
        )

        n_1,n_0 = len(X1),len(X0)

        sign, logdet = np.linalg.slogdet(Sigma_1+self.min_var*np.eye(Sigma_1.shape[0]))
        return 0.5*(logdet - n_0 * np.log(self.model.model.likelihood[0]+self.min_var*np.eye(Sigma_1.shape[0])) - n_1 * np.log(self.model.model.likelihood[1]+self.min_var*np.eye(Sigma_1.shape[0])))
    
    def causal_EIG_closed_form(self,X,T,holdout_X = None):

        if holdout_X is None:
            holdout_X0 = self.X0_train
            holdout_X1 = self.X1_train
        
        else:
            holdout_X = holdout_X

            holdout_X0 = np.array(np.hstack([holdout_X, np.zeros_like(holdout_X[:, 1].reshape((len(holdout_X[:, 1]), 1)))]))
            holdout_X1 = np.hstack([holdout_X, np.ones_like(holdout_X[:, 1].reshape((len(holdout_X[:, 1]), 1)))])
            
        
        X0 = X[T==0]
        X0 = np.array(
                np.hstack([X0, np.zeros_like(X0[:, 1].reshape((len(X0[:, 1]), 1)))])
            )
        X1 = X[T==1]
        X1 = np.array(
            np.hstack([X1, np.ones_like(X1[:, 1].reshape((len(X1[:, 1]), 1)))])
            )
        
        X0_shape = X0.shape
        X1_shape = X1.shape

        noise_dict_0 = {
            "output_index": X0[:, X0_shape[1] - 1]
            .reshape((X0_shape[0], 1))
            .astype(int)}
        
        noise_dict_1 = {
            "output_index": X1[:, X1_shape[1] - 1]
            .reshape((X1_shape[0], 1))
            .astype(int)}
        
        noise_dict_0_train = {
            "output_index": holdout_X0[:, holdout_X0.shape[1] - 1]
            .reshape((holdout_X0.shape[0], 1))
            .astype(int)}

        noise_dict_1_train = {
            "output_index": holdout_X1[:, holdout_X1.shape[1] - 1]
            .reshape((holdout_X1.shape[0], 1))
            .astype(int)}
        
        Sigma_1 = np.block(
            [[self.model.model.posterior_covariance_between_points(X0,X0,Y_metadata=noise_dict_0), self.model.model.posterior_covariance_between_points(X0,X1,include_likelihood=False) ],
            [self.model.model.posterior_covariance_between_points(X1,X0,include_likelihood=False), self.model.model.posterior_covariance_between_points(X1,X1,Y_metadata=noise_dict_1) ]]
        )

        Sigma_2 = self.model.model.posterior_covariance_between_points(holdout_X0,holdout_X0,Y_metadata=noise_dict_0_train)+self.model.model.posterior_covariance_between_points(holdout_X1,holdout_X1,Y_metadata=noise_dict_1_train)-2*self.model.model.posterior_covariance_between_points(holdout_X0,holdout_X1,include_likelihood=False)

        Sigma_join = np.concatenate([
            self.model.model.posterior_covariance_between_points(holdout_X1,X0,include_likelihood=False)-self.model.model.posterior_covariance_between_points(holdout_X0,X0,include_likelihood=False),
            self.model.model.posterior_covariance_between_points(holdout_X1,X1,include_likelihood=False)-self.model.model.posterior_covariance_between_points(holdout_X0,X1,include_likelihood=False)
                                     ],axis=1)
        
        Sigma = np.block([
            [Sigma_1, Sigma_join.T],
            [Sigma_join, Sigma_2]
        ]

        )
    

        sign, logdet1 = np.linalg.slogdet(Sigma_1+self.min_var*np.eye(Sigma_1.shape[0]))
        sign, logdet2 = np.linalg.slogdet(Sigma_2+self.min_var*np.eye(Sigma_2.shape[0]))
        sign, logdet_sig = np.linalg.slogdet(Sigma+self.min_var*np.eye(Sigma.shape[0]))

        return logdet1+logdet2-logdet_sig
