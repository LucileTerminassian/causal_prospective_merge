import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *


############################ CLOSED FORM


def linear_eig_closed_form_varying_sample_size(
    data: dict[int, dict],
    data_parameters: dict[str, Any],
    prior_hyperparameters: dict[str, Any],  # passed to BayesianLinearRegression
    verbose: bool = True,
):
    # this now works with both exact and "non-exact" `data`
    sample_sizes = data_parameters["varying_sample_sizes"]
    candidates_names = data[sample_sizes[0]].keys() - ["host"]
    
    # dict of the form {candidate_name: {sample_size: EIG}}
    EIG_obs = {name: [] for name in candidates_names}
    EIG_caus = {name: [] for name in candidates_names}

    for length in sample_sizes:
        dlen = data[length]  # for convenience
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(data_parameters["causal_param_first_index"])

        ### Bayesian update on host data using closed form
        X_host = torch.from_numpy(dlen["host"].drop(columns=["Y"]).values)
        Y_host = torch.from_numpy(dlen["host"]["Y"].values)
        # fit the posterior (updates the params in the model class)
        bayes_reg.fit(X_host, Y_host)

        if verbose:
            print(f"For a sample size of {length}")
            print(f" % treated in host: {int(100 * dlen['host']['T'].mean())}%")

        for cand in candidates_names:
            X_cand = torch.from_numpy(dlen[cand].drop(columns=["Y"]).values)
            if verbose:
                print(f" % treated in {cand}: {int(100 * dlen[cand]['T'].mean())}%")

            EIG_obs[cand].append(bayes_reg.closed_form_obs_EIG(X_cand))
            EIG_caus[cand].append(bayes_reg.closed_form_causal_EIG(X_cand))

    return EIG_obs, EIG_caus


#################################### FROM SAMPLES

def linear_eig_from_samples_varying_sample_size(
    data: dict[int, dict],
    data_parameters: dict[str, Any],
    prior_hyperparameters: dict[str, Any],
    sampling_parameters: dict[str, int],    
    verbose: bool = False,
):
    sample_sizes = data_parameters["varying_sample_sizes"]
    candidates_names = data[sample_sizes[0]].keys() - ["host"]
    (
        n_samples_outer_expectation_obs,
        n_samples_inner_expectation_obs,
        n_samples_inner_expectation_caus,
        n_samples_outer_expectation_caus,
    ) = (
        sampling_parameters["n_samples_outer_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_caus"],
        sampling_parameters["n_samples_outer_expectation_caus"],
    )

    # dict of the form {candidate_name: {sample_size: EIG}}
    EIG_obs = {name: [] for name in candidates_names}
    EIG_caus = {name: [] for name in candidates_names}

    for length in sample_sizes:
        dlen = data[length]  # for convenience
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(data_parameters["causal_param_first_index"])

        ### Bayesian update on host data using closed form
        X_host = torch.from_numpy(dlen["host"].drop(columns=["Y"]).values)
        Y_host = torch.from_numpy(dlen["host"]["Y"].values)
        # fit the posterior (updates the params in the model class)
        bayes_reg.fit(X_host, Y_host)
        

        if verbose:
            print(f"For a sample size of {length}")
            print(f" % treated in host: {int(100 * dlen['host']['T'].mean())}%")

        for cand in candidates_names:
            X_cand = torch.from_numpy(dlen[cand].drop(columns=["Y"]).values)
            if verbose:
                print(f" % treated in {cand}: {int(100 * dlen[cand]['T'].mean())}%")

            EIG_obs[cand].append(bayes_reg.samples_obs_EIG(
                X_cand, n_samples_outer_expectation_obs, n_samples_inner_expectation_obs))
            EIG_caus[cand].append(bayes_reg.samples_causal_EIG(
                X_cand, n_samples_outer_expectation_caus, n_samples_inner_expectation_caus))

    return EIG_obs, EIG_caus


def bart_eig_from_samples_varying_sample_size(
    
    data: dict[int, dict],
    data_parameters: dict[str, Any],
    prior_hyperparameters: dict[str, Any],
    predictive_model_parameters: dict[str, int],
    conditional_model_param: dict[str, int],
    sampling_parameters: dict[str, int],
    verbose: bool =False):

    sample_sizes = data_parameters["varying_sample_sizes"]
    candidates_names = data[sample_sizes[0]].keys() - ["host"]

    (
        n_samples_outer_expectation_obs,
        n_samples_inner_expectation_obs,
        n_samples_inner_expectation_caus,
        n_samples_outer_expectation_caus,
    ) = (
        sampling_parameters["n_samples_outer_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_caus"],
        sampling_parameters["n_samples_outer_expectation_caus"],
    )

    # dict of the form {candidate_name: {sample_size: EIG}}
    EIG_obs = {name: [] for name in candidates_names}
    EIG_caus = {name: [] for name in candidates_names}

    for length in sample_sizes:
        dlen = data[length]  # for convenience

        ### Bayesian update on host data using closed form
        X_host_not_T = dlen["host"].drop(columns=["Y","T"]).values
        T_host = dlen["host"]["T"].values.astype(np.int32)    
        Y_host = dlen["host"]["Y"].values
        # fit the posterior (updates the params in the model class)
        bcf = BayesianCausalForest(
            prior_hyperparameters,
            predictive_model_parameters=predictive_model_parameters,
            conditional_model_param=conditional_model_param,
        )

        bcf.store_train_data(X=X_host_not_T, T=T_host, Y=Y_host)

        if verbose:
            print(f"For a sample size of {length}")
            print(f" % treated in host: {int(100 * dlen['host']['T'].mean())}%")

        for cand in candidates_names:
            X_cand_not_T = dlen[cand].drop(columns=["Y","T"]).values
            T_cand = dlen[cand]["T"].values.astype(np.int32)

            if verbose:
                print(f" % treated in {cand}: {int(100 * dlen[cand]['T'].mean())}%")

            joint_eig = bcf.joint_EIG_calc(X_cand_not_T, \
                                T_cand, sampling_parameters)

            EIG_obs[cand].append(joint_eig["Obs EIG"])
            EIG_caus[cand].append(joint_eig["Causal EIG"])

    return EIG_obs, EIG_caus



