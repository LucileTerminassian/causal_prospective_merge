import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *


############################ CLOSED FORM

<<<<<<< HEAD

def linear_eig_closed_form_varying_sample_size(
    data: dict[int, dict],
    data_parameters: dict[str, Any],
    prior_hyperparameters: dict[str, Any],  # passed to BayesianLinearRegression
    verbose: bool = True,
):
    # this now works with both exact and "non-exact" `data`
    sample_sizes = data_parameters["n_both_candidates_list"]
    condidates_names = data[sample_sizes[0]].keys() - ["host"]
    # dict of the form {candidate_name: {sample_size: EIG}}
    EIG_obs = {name: [] for name in condidates_names}
    EIG_caus = {name: [] for name in condidates_names}
=======
## noisy


def linear_eig_closed_form_varying_sample_size(
    data, data_parameters, sigma_rand_error, prior_hyperparameters, n_mc, verbose=True
):

    n_both_candidates_list, proportion = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    std_true_y, causal_param_first_index = (
        data_parameters["std_true_y"],
        data_parameters["causal_param_first_index"],
    )

    results = {
        "EIG_obs_closed_form_complementary": [],
        "EIG_obs_closed_form_cand2": [],
        "EIG_caus_closed_form_complementary": [],
        "EIG_caus_closed_form_cand2": [],
    }

    for length in n_both_candidates_list:

        n_both_candidates_list, proportion, causal_param_first_index = (
            data_parameters["n_both_candidates_list"],
            data_parameters["proportion"],
            data_parameters["causal_param_first_index"],
        )
>>>>>>> main

    for length in sample_sizes:
        dlen = data[length]  # for convenience
        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(data_parameters["causal_param_first_index"])

        ### Bayesian update on host data using closed form
<<<<<<< HEAD
        X_host = torch.from_numpy(dlen["host"].drop(columns=["Y"]).values)
        Y_host = torch.from_numpy(dlen["host"]["Y"].values)
        # fit the posterior (updates the params in the model class)
        bayes_reg.fit(X_host, Y_host)

        if verbose:
            print(f"For a sample size of {length}")
            print(f" % treated in host: {int(100 * dlen['host']['T'].mean())}%")

        for cand in condidates_names:
            X_cand = torch.from_numpy(dlen[cand].drop(columns=["Y"]).values)
            if verbose:
                print(f" % treated in {cand}: {int(100 * dlen[cand]['T'].mean())}%")

            EIG_obs[cand].append(bayes_reg.closed_form_obs_EIG(X_cand))
            EIG_caus[cand].append(bayes_reg.closed_form_causal_EIG(X_cand))

    return EIG_obs, EIG_caus
=======
        X_host, Y_host = torch.from_numpy(
            data[length]["host"].drop(columns=["Y"]).values
        ), torch.from_numpy(data[length]["host"]["Y"].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        X_complementary = torch.from_numpy(
            data[length]["complementary"].drop(columns=["Y"]).values
        )
        X_cand2 = torch.from_numpy(data[length]["cand2"].drop(columns=["Y"]).values)

        if verbose:

            T_host, T_complementary, T_cand2 = (
                data[length]["host"]["T"],
                data[length]["complementary"]["T"],
                data[length]["cand2"]["T"],
            )

            percent_treated_host = 100 * sum(T_host) / len(T_host)
            percent_treated_complementaryr = (
                100 * sum(T_complementary) / len(T_complementary)
            )
            percent_treated_cand2 = 100 * sum(T_cand2) / len(T_cand2)

            print("For a sample size in complementary and host of " + str(length))
            print(f"Percentage of treated in host: {percent_treated_host}%")
            print(
                f"Percentage of treated in complementary: {percent_treated_complementaryr}%"
            )
            print(f"Percentage of treated in cand2: {percent_treated_cand2}%")

        results["EIG_obs_closed_form_complementary"].append(
            bayes_reg.closed_form_obs_EIG(X_complementary)
        )
        results["EIG_obs_closed_form_cand2"].append(
            bayes_reg.closed_form_obs_EIG(X_cand2)
        )

        results["EIG_caus_closed_form_complementary"].append(
            bayes_reg.closed_form_causal_EIG(X_complementary)
        )
        results["EIG_caus_closed_form_cand2"].append(
            bayes_reg.closed_form_causal_EIG(X_cand2)
        )

    EIG_obs_closed_form = np.vstack(
        (
            results["EIG_obs_closed_form_complementary"],
            results["EIG_obs_closed_form_cand2"],
        )
    )
    EIG_caus_closed_form = np.vstack(
        (
            results["EIG_caus_closed_form_complementary"],
            results["EIG_caus_closed_form_cand2"],
        )
    )

    return EIG_obs_closed_form, EIG_caus_closed_form


## exact


def linear_eig_closed_form_exact_datasets(
    data, data_parameters, sigma_rand_error, prior_hyperparameters, n_mc
):

    n_both_candidates_list, proportion = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    std_true_y, causal_param_first_index = (
        data_parameters["std_true_y"],
        data_parameters["causal_param_first_index"],
    )

    dict_results_obs = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }
    dict_results_caus = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }

    for length in n_both_candidates_list:

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        ### Bayesian update on host data using closed form
        X_host, Y_host = torch.from_numpy(
            data[length]["host"].drop(columns=["Y"]).values
        ), torch.from_numpy(data[length]["host"]["Y"].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        X_exact_complementary = torch.from_numpy(
            data[length]["exact_complementary"].drop(columns=["Y"]).values
        )
        X_exact_twin = torch.from_numpy(
            data[length]["exact_twin"].drop(columns=["Y"]).values
        )
        X_exact_twin_treated = torch.from_numpy(
            data[length]["exact_twin_treated"].drop(columns=["Y"]).values
        )
        X_exact_twin_untreated = torch.from_numpy(
            data[length]["exact_twin_untreated"].drop(columns=["Y"]).values
        )

        dict_results_obs["complementary"].append(
            bayes_reg.closed_form_obs_EIG(X_exact_complementary)
        )
        dict_results_obs["twin"].append(bayes_reg.closed_form_obs_EIG(X_exact_twin))
        dict_results_obs["twin_treated"].append(
            bayes_reg.closed_form_obs_EIG(X_exact_twin_treated)
        )
        dict_results_obs["twin_untreated"].append(
            bayes_reg.closed_form_obs_EIG(X_exact_twin_untreated)
        )

        dict_results_caus["complementary"].append(
            bayes_reg.closed_form_causal_EIG(X_exact_complementary)
        )
        dict_results_caus["twin"].append(bayes_reg.closed_form_causal_EIG(X_exact_twin))
        dict_results_caus["twin_treated"].append(
            bayes_reg.closed_form_causal_EIG(X_exact_twin_treated)
        )
        dict_results_caus["twin_untreated"].append(
            bayes_reg.closed_form_causal_EIG(X_exact_twin_untreated)
        )

    return dict_results_obs, dict_results_caus
>>>>>>> main


#################################### FROM SAMPLES
## noisy


<<<<<<< HEAD
# What's the difference with the above?
=======
>>>>>>> main
def linear_eig_from_samples_varying_sample_size(
    data, data_parameters, prior_hyperparameters, sampling_parameters
):

    results = {
<<<<<<< HEAD
        "EIG_obs_from_samples_mirror": [],
        "EIG_obs_from_samples_cand2": [],
        "EIG_caus_from_samples_mirror": [],
=======
        "EIG_obs_from_samples_complementary": [],
        "EIG_obs_from_samples_cand2": [],
        "EIG_caus_from_samples_complementary": [],
>>>>>>> main
        "EIG_caus_from_samples_cand2": [],
    }

    (
        n_both_candidates_list,
        proportion,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    causal_param_first_index = data_parameters["causal_param_first_index"]

    (
        n_samples_outer_expectation,
        n_samples_inner_expectation,
        n_causal_inner_exp,
        n_causal_outer_exp,
    ) = (
        sampling_parameters["n_samples_outer_expectation"],
        sampling_parameters["n_samples_inner_expectation"],
        sampling_parameters["n_causal_inner_exp"],
        sampling_parameters["n_causal_outer_exp"],
    )

    for length in n_both_candidates_list:

        X_host, Y_host = torch.from_numpy(
            data[length]["host"].drop(columns=["Y"]).values
        ), torch.from_numpy(data[length]["host"]["Y"].values)
<<<<<<< HEAD
        X_mirror = torch.from_numpy(data[length]["mirror"].drop(columns=["Y"]).values)
=======
        X_complementary = torch.from_numpy(
            data[length]["complementary"].drop(columns=["Y"]).values
        )
>>>>>>> main
        X_cand2 = torch.from_numpy(data[length]["cand2"].drop(columns=["Y"]).values)

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        n_samples = n_samples_outer_expectation * (n_samples_inner_expectation + 1)
        beta_post_host_samples = bayes_reg.posterior_sample(n_samples=n_samples)

<<<<<<< HEAD
        results["EIG_obs_from_samples_mirror"].append(
            bayes_reg.samples_obs_EIG(
                X_mirror, n_samples_outer_expectation, n_samples_inner_expectation
=======
        results["EIG_obs_from_samples_complementary"].append(
            bayes_reg.samples_obs_EIG(
                X_complementary,
                n_samples_outer_expectation,
                n_samples_inner_expectation,
>>>>>>> main
            )
        )
        results["EIG_obs_from_samples_cand2"].append(
            bayes_reg.samples_obs_EIG(
                X_cand2, n_samples_outer_expectation, n_samples_inner_expectation
            )
        )

<<<<<<< HEAD
        results["EIG_caus_from_samples_mirror"].append(
            bayes_reg.samples_causal_EIG(
                X_mirror, n_causal_outer_exp, n_causal_inner_exp
=======
        results["EIG_caus_from_samples_complementary"].append(
            bayes_reg.samples_causal_EIG(
                X_complementary, n_causal_outer_exp, n_causal_inner_exp
>>>>>>> main
            )
        )
        results["EIG_caus_from_samples_cand2"].append(
            bayes_reg.samples_causal_EIG(
                X_cand2, n_causal_outer_exp, n_causal_inner_exp
            )
        )

    EIG_obs_samples = np.vstack(
<<<<<<< HEAD
        (results["EIG_obs_from_samples_mirror"], results["EIG_obs_from_samples_cand2"])
    )
    EIG_caus_samples = np.vstack(
        (
            results["EIG_caus_from_samples_mirror"],
=======
        (
            results["EIG_obs_from_samples_complementary"],
            results["EIG_obs_from_samples_cand2"],
        )
    )
    EIG_caus_samples = np.vstack(
        (
            results["EIG_caus_from_samples_complementary"],
>>>>>>> main
            results["EIG_caus_from_samples_cand2"],
        )
    )

    return EIG_obs_samples, EIG_caus_samples


## exact


def linear_eig_from_samples_exact_datasets(
    data, data_parameters, prior_hyperparameters, n_mc, sampling_parameters
):

    (
        n_both_candidates_list,
        proportion,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    causal_param_first_index = data_parameters["causal_param_first_index"]

    (
        n_samples_outer_expectation,
        n_samples_inner_expectation,
        n_causal_inner_exp,
        n_causal_outer_exp,
    ) = (
        sampling_parameters["n_samples_outer_expectation"],
        sampling_parameters["n_samples_inner_expectation"],
        sampling_parameters["n_causal_inner_exp"],
        sampling_parameters["n_causal_outer_exp"],
    )

    dict_results_obs = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }
    dict_results_caus = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }

    for length in n_both_candidates_list:

        bayes_reg = BayesianLinearRegression(prior_hyperparameters)
        bayes_reg.set_causal_index(causal_param_first_index)

        ### Bayesian update on host data using closed form
        X_host, Y_host = torch.from_numpy(
            data[length]["host"].drop(columns=["Y"]).values
        ), torch.from_numpy(data[length]["host"]["Y"].values)
        post_host_parameters = bayes_reg.fit(X_host, Y_host)

        n_samples = n_samples_outer_expectation * (n_samples_inner_expectation + 1)
        beta_post_host_samples = bayes_reg.posterior_sample(n_samples=n_samples)

        X_exact_complementary = torch.from_numpy(
            data[length]["exact_complementary"].drop(columns=["Y"]).values
        )
        X_exact_twin = torch.from_numpy(
            data[length]["exact_twin"].drop(columns=["Y"]).values
        )
        X_exact_twin_treated = torch.from_numpy(
            data[length]["exact_twin_treated"].drop(columns=["Y"]).values
        )
        X_exact_twin_untreated = torch.from_numpy(
            data[length]["exact_twin_untreated"].drop(columns=["Y"]).values
        )

        dict_results_obs["complementary"].append(
            bayes_reg.samples_obs_EIG(
                X_exact_complementary,
                n_samples_outer_expectation,
                n_samples_inner_expectation,
            )
        )
        dict_results_obs["twin"].append(
            bayes_reg.samples_obs_EIG(
                X_exact_twin, n_samples_outer_expectation, n_samples_inner_expectation
            )
        )
        dict_results_obs["twin_treated"].append(
            bayes_reg.samples_obs_EIG(
                X_exact_twin_treated,
                n_samples_outer_expectation,
                n_samples_inner_expectation,
            )
        )
        dict_results_obs["twin_untreated"].append(
            bayes_reg.samples_obs_EIG(
                X_exact_twin_untreated,
                n_samples_outer_expectation,
                n_samples_inner_expectation,
            )
        )

        dict_results_caus["complementary"].append(
            bayes_reg.samples_causal_EIG(
                X_exact_complementary, n_causal_outer_exp, n_causal_inner_exp
            )
        )
        dict_results_caus["twin"].append(
            bayes_reg.samples_causal_EIG(
                X_exact_twin, n_causal_outer_exp, n_causal_inner_exp
            )
        )
        dict_results_caus["twin_treated"].append(
            bayes_reg.samples_causal_EIG(
                X_exact_twin_treated, n_causal_outer_exp, n_causal_inner_exp
            )
        )
        dict_results_caus["twin_untreated"].append(
            bayes_reg.samples_causal_EIG(
                X_exact_twin_untreated, n_causal_outer_exp, n_causal_inner_exp
            )
        )

    return dict_results_obs, dict_results_caus


#################################### FROM SAMPLES

## noisy


def bart_eig_from_samples_varying_sample_size(
    data,
    data_parameters,
    prior_hyperparameters,
    predictive_model_parameters,
    conditional_model_param,
    sampling_parameters,
):

    # data is nested dictionary
    # data[length]['host] is a pandas df for the host site

    results = {
<<<<<<< HEAD
        "EIG_obs_from_samples_mirror": [],
        "EIG_obs_from_samples_cand2": [],
        "EIG_caus_from_samples_mirror": [],
=======
        "EIG_obs_from_samples_complementary": [],
        "EIG_obs_from_samples_cand2": [],
        "EIG_caus_from_samples_complementary": [],
>>>>>>> main
        "EIG_caus_from_samples_cand2": [],
    }

    (
        n_both_candidates_list,
        proportion,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    causal_param_first_index = data_parameters["causal_param_first_index"]

    (
        n_samples_outer_expectation,
        n_samples_inner_expectation,
        n_causal_inner_exp,
        n_causal_outer_exp,
    ) = (
<<<<<<< HEAD
        sampling_parameters["n_samples_outer_expectation"],
        sampling_parameters["n_samples_inner_expectation"],
        sampling_parameters["n_causal_inner_exp"],
        sampling_parameters["n_causal_outer_exp"],
=======
        sampling_parameters["n_samples_outer_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_obs"],
        sampling_parameters["n_samples_inner_expectation_caus"],
        sampling_parameters["n_samples_outer_expectation_caus"],
>>>>>>> main
    )

    for length in n_both_candidates_list:

        X_host, T_host, Y_host = (
            data[length]["host"].drop(columns=["Y", "T"]).values,
            data[length]["host"]["T"].values,
            data[length]["host"]["Y"].values,
        )

<<<<<<< HEAD
        X_mirror, T_mirror = (
            data[length]["mirror"].drop(columns=["Y", "T"]).values,
            data[length]["mirror"]["T"].values,
        )
        X_cand2, T_cand2 = (
            data[length]["cand2"].drop(columns=["Y", "T"]).values,
            data[length]["mirror"]["T"].values,
=======
        X_complementary, T_complementary = (
            data[length]["complementary"].drop(columns=["Y", "T"]).values,
            data[length]["complementary"]["T"].values,
        )
        X_cand2, T_cand2 = (
            data[length]["cand2"].drop(columns=["Y", "T"]).values,
            data[length]["cand2"]["T"].values,
        )

        T_host, T_complementary, T_cand2 = (
            T_host.astype(np.int32),
            T_complementary.astype(np.int32),
            T_cand2.astype(np.int32),
>>>>>>> main
        )

        bcf = BayesianCausalForest(
            prior_hyperparameters,
            predictive_model_parameters={"num_trees_pr": 200, "num_trees_trt": 100},
            conditional_model_param={"num_trees_pr": 200},
        )
        bcf.store_train_data(X=X_host, T=T_host, Y=Y_host)

<<<<<<< HEAD
        results_mirror = bcf.joint_EIG_calc(X_mirror, T_mirror, sampling_parameters)

        results_cand2 = bcf.joint_EIG_calc(X_cand2, T_cand2, sampling_parameters)

        results["EIG_obs_from_samples_mirror"].append(results_mirror["Obs EIG"])
        results["EIG_obs_from_samples_cand2"].append(results_cand2["Obs EIG"])

        results["EIG_caus_from_samples_mirror"].append(results_mirror["Causal EIG"])
        results["EIG_caus_from_samples_cand2"].append(results_cand2["Causal EIG"])

    EIG_obs_samples = np.vstack(
        (results["EIG_obs_from_samples_mirror"], results["EIG_obs_from_samples_cand2"])
    )
    EIG_caus_samples = np.vstack(
        (
            results["EIG_caus_from_samples_mirror"],
=======
        results_complementary = bcf.joint_EIG_calc(
            X_complementary, T_complementary, sampling_parameters
        )

        results_cand2 = bcf.joint_EIG_calc(X_cand2, T_cand2, sampling_parameters)

        results["EIG_obs_from_samples_complementary"].append(
            results_complementary["Obs EIG"]
        )
        results["EIG_obs_from_samples_cand2"].append(results_cand2["Obs EIG"])

        results["EIG_caus_from_samples_complementary"].append(
            results_complementary["Causal EIG"]
        )
        results["EIG_caus_from_samples_cand2"].append(results_cand2["Causal EIG"])

    EIG_obs_samples = np.vstack(
        (
            results["EIG_obs_from_samples_complementary"],
            results["EIG_obs_from_samples_cand2"],
        )
    )
    EIG_caus_samples = np.vstack(
        (
            results["EIG_caus_from_samples_complementary"],
>>>>>>> main
            results["EIG_caus_from_samples_cand2"],
        )
    )

    return EIG_obs_samples, EIG_caus_samples


## exact


def bart_eig_from_samples_exact_datasets(
    data,
    data_parameters,
    prior_hyperparameters,
    predictive_model_parameters,
    conditional_model_param,
    sampling_parameters,
):

    (
        n_both_candidates_list,
        proportion,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
    )
    causal_param_first_index = data_parameters["causal_param_first_index"]

    (
        n_samples_outer_expectation,
        n_samples_inner_expectation,
        n_causal_inner_exp,
        n_causal_outer_exp,
    ) = (
        sampling_parameters["n_samples_outer_expectation"],
        sampling_parameters["n_samples_inner_expectation"],
        sampling_parameters["n_causal_inner_exp"],
        sampling_parameters["n_causal_outer_exp"],
    )

    dict_results_obs = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }
    dict_results_caus = {
        "complementary": [],
        "twin": [],
        "twin_treated": [],
        "twin_untreated": [],
    }

    for length in n_both_candidates_list:

        X_exact_complementary, T_exact_complementary = (
            data[length]["exact_complementary"].drop(columns=["Y", "T"]).values,
<<<<<<< HEAD
            data[length]["host"]["T"].values,
=======
            data[length]["exact_complementary"]["T"].values,
>>>>>>> main
        )
        X_exact_twin, T_exact_twin = (
            data[length]["exact_twin"].drop(columns=["Y", "T"]).values,
            data[length]["exact_twin"]["T"].values,
        )
        X_exact_twin_treated, T_exact_twin_treated = (
            data[length]["exact_twin_treated"].drop(columns=["Y", "T"]).values,
            data[length]["exact_twin_treated"]["T"].values,
        )
        X_exact_twin_untreated, T_exact_twin_untreated = (
            data[length]["exact_twin_untreated"].drop(columns=["Y", "T"]).values,
            data[length]["exact_twin_untreated"]["T"].values,
        )

        X_host, T_host, Y_host = (
            data[length]["host"].drop(columns=["Y", "T"]).values,
            data[length]["host"]["T"].values,
            data[length]["host"]["Y"].values,
        )

        bcf = BayesianCausalForest(
            prior_hyperparameters,
            predictive_model_parameters=predictive_model_parameters,
            conditional_model_param=conditional_model_param,
        )
        bcf.store_train_data(X=X_host, T=T_host, Y=Y_host)

        results_complementary = bcf.joint_EIG_calc(
            X_exact_complementary, T_exact_complementary, sampling_parameters
        )

        results_twin = bcf.joint_EIG_calc(
            X_exact_twin, T_exact_twin, sampling_parameters
        )

        results_twin_treated = bcf.joint_EIG_calc(
            X_exact_twin_treated, T_exact_twin_treated, sampling_parameters
        )

        results_twin_untreated = bcf.joint_EIG_calc(
            X_exact_twin_untreated, T_exact_twin_untreated, sampling_parameters
        )

        dict_results_obs["complementary"].append(results_complementary["Obs EIG"])
        dict_results_obs["twin"].append(results_twin["Obs EIG"])
        dict_results_obs["twin_treated"].append(results_twin_treated["Obs EIG"])
        dict_results_obs["twin_untreated"].append(results_twin_untreated["Obs EIG"])

        dict_results_obs["complementary"].append(results_complementary["Causal EIG"])
        dict_results_obs["twin"].append(results_twin["Causal EIG"])
        dict_results_obs["twin_treated"].append(results_twin_treated["Causal EIG"])
        dict_results_obs["twin_untreated"].append(results_twin_untreated["Causal EIG"])

    return dict_results_obs, dict_results_caus


if __name__ == "__main__":
    from rct_data_generator import _main

    _, data_parameters = _main()
    exact_data = generate_exact_synthetic_data_varying_sample_size(data_parameters)

    sigma_prior = 1.0
    sigma_rand_error = 1.0
    prior_mean = np.array([0, 1, 0, 0, 1, 0, 0, 0])

    # Number of samples used to estimate outer expectation
    n_samples_for_expectation = 50
    m_samples_for_expectation = 1000

    beta_0, sigma_0_sq, inv_cov_0 = (
        prior_mean,
        sigma_rand_error**2,
        1 / sigma_prior * np.eye(len(prior_mean)),
    )
    prior_hyperparameters = {
        "beta_0": beta_0,
        "sigma_0_sq": sigma_0_sq,
        "inv_cov_0": inv_cov_0,
    }

    EIGs = linear_eig_closed_form_varying_sample_size(  # CHECK what this does
        exact_data,
        data_parameters,
        prior_hyperparameters,
        verbose=True,
    )
    print(EIGs)
