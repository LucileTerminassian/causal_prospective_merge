import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
import sys


def generate_rct(x_distributions: dict[str, np.ndarray], seed=0):
    np.random.seed(seed)
    X = pd.DataFrame.from_dict(x_distributions)
    n_global = X.shape[0]
    T = np.random.randint(0, 2, size=n_global)  # Generate T
    return X, T


# Probability functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to generate host and complementary data
def subsample_two_complementary_datasets(
    X,
    T,
    f_assigned_to_host,
    n_host,
    n_complementary,
    power_x,
    power_x_t,
    outcome_function,
    std_true_y,
    seed=0,
):
    np.random.seed(seed)
    n_global = np.shape(X)[0]
    # Initialize dictionaries
    data_host, data_complementary = {}, {}

    # Add 'T' key to each dictionary
    for name in X.columns:
        data_host[name] = []
        data_complementary[name] = []
    # Add 'T' key to each dictionary
    data_host["T"] = []
    data_complementary["T"] = []

    if n_host + n_complementary > n_global:
        print("n_host + n_complementary > n_rct")
        return

    done_complementary, done_host = False, False

    for i in range(n_global):

        proba_assigned_to_host = f_assigned_to_host(
            X.iloc[i, :], T[i], np.random.normal()
        )
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)

        first_column_host = next(iter(data_host.values()))
        first_column_complementary = next(iter(data_complementary.values()))

        if is_assigned_to_host:
            if len(first_column_host) < n_host:
                for column_name in X.columns:
                    data_host[column_name].append(X.iloc[i][column_name])
                data_host["T"].append(T[i])
            else:
                done_host = True
                pass

        else:
            if len(first_column_complementary) < n_complementary:
                for column_name in X.columns:
                    data_complementary[column_name].append(X.iloc[i][column_name])
                data_complementary["T"].append(T[i])
            else:
                done_complementary = True
                pass
        if done_complementary and done_host:
            break

    data_host = pd.DataFrame.from_dict(data_host)
    data_complementary = pd.DataFrame.from_dict(data_complementary)

    if len(data_complementary) != n_complementary:
        print(
            "len(data_complementary) n="
            + str(len(data_complementary))
            + " != n_complementary ("
            + str(n_complementary)
            + ")"
        )
    if len(data_host) != n_host:
        print(
            "len(data_host) n="
            + str(len(data_host))
            + " != n_complementary ("
            + str(n_host)
            + ")"
        )

    design_data_host = generate_design_matrix(data_host, power_x, power_x_t)
    design_data_complementary = generate_design_matrix(
        data_complementary, power_x, power_x_t
    )

    design_data_host = add_outcome(design_data_host, outcome_function, std_true_y)
    design_data_complementary = add_outcome(
        design_data_complementary, outcome_function, std_true_y
    )

    return design_data_host, design_data_complementary


def subsample_one_dataset(
    X,
    T,
    assignment_function,
    sample_size,
    power_x,
    power_x_t,
    outcome_function,
    std_true_y,
    seed=0,
):

    np.random.seed(seed)
    n_global = np.shape(X)[0]
    data = {}
    for name in X.columns:
        data[name] = []

    # Add 'T' key to each dictionary
    data["T"] = []

    if sample_size > n_global:
        print("n_cand2 > n_rct")
        return

    for i in range(n_global):
        proba_assigned = assignment_function(X.iloc[i, :], T[i], np.random.normal())
        selected = np.random.binomial(1, proba_assigned)
        if selected == 1:
            first_value = next(iter(data.values()))
            if len(first_value) < sample_size:
                for column_name in X.columns:
                    data[column_name].append(X.iloc[i][column_name])
                data["T"].append(T[i])
            else:
                break

    data = pd.DataFrame.from_dict(data)

    if len(data) != sample_size:
        print(
            "len(data_cand2) n="
            + str(len(data))
            + " != n_complementary ("
            + str(sample_size)
            + ")"
        )

    design_data = generate_design_matrix(data, power_x, power_x_t)
    design_data = add_outcome(design_data, outcome_function, std_true_y)

    return design_data


def generate_design_matrix(data, power_x, power_x_t):
    # Extract X and T from the dataframe
    X = data.drop(columns=["T"])
    T = data["T"]

    # Initialize a dataframe to hold the design matrix with intercept column filled with ones
    n, d = np.shape(X)
    X_prime = pd.DataFrame(np.ones((n, d * power_x + d * power_x_t + 2)))

    # Create a list to hold column names
    column_names = ["intercept"]

    for i in range(1, power_x + 1):
        for col in X.columns:
            if i > 1:
                column_names.append(f"{col}**{i}")
            else:
                column_names.append(f"{col}")

    column_names.append("T")

    for i in range(1, power_x_t + 1):
        for col in X.columns:
            if i > 1:
                column_names.append(f"T*{col}**{i}")
            else:
                column_names.append(f"T*{col}")

    # Set column names for X_prime
    X_prime.columns = column_names

    # Concatenate X^i for i = 1 to power_x
    for i in range(1, power_x + 1):
        for col in X.columns:
            if i > 1:
                X_prime[f"{col}**{i}"] = X[col] ** i
            else:
                X_prime[f"{col}"] = X[col] ** i

    X_prime["T"] = T

    # Concatenate T*X^i for i = 1 to power_x_t
    for i in range(1, power_x_t + 1):
        for col in X.columns:
            if i > 1:
                X_prime[f"T*{col}**{i}"] = T * (X[col] ** i)
            else:
                X_prime[f"T*{col}"] = T * (X[col] ** i)

    return X_prime


def add_outcome(data, outcome_function, scale):

    n = np.shape(data)[0]
    X = data.drop(columns=["T"])
    T = data["T"]
    eps = np.random.normal(size=n, scale=scale)

    Y = outcome_function(X, T, eps)
    data["Y"] = Y

    return data


def generate_data_varying_sample_size(
    data_parameters, x_distributions=None, X=None, T=None
):

    (
        n_both_candidates_list,
        proportion,
        x_distributions,
        p_assigned_to_cand2,
        p_assigned_to_host,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
        data_parameters["x_distributions"],
        data_parameters["p_assigned_to_cand2"],
        data_parameters["p_assigned_to_host"],
    )
    (
        n_host,
        power_x,
        power_x_t,
        outcome_function,
        std_true_y,
        causal_param_first_index,
    ) = (
        data_parameters["n_host"],
        data_parameters["power_x"],
        data_parameters["power_x_t"],
        data_parameters["outcome_function"],
        data_parameters["std_true_y"],
        data_parameters["causal_param_first_index"],
    )

    data = {}

    for seed, length in enumerate(n_both_candidates_list):
        if X is None:  # synthetic
            X, T = generate_rct(x_distributions, seed=seed)
            pre_X_cand2, pre_T_cand2 = generate_rct(x_distributions, seed=seed)

        design_data_host, design_data_complementary = (
            subsample_two_complementary_datasets(
                X,
                T,
                p_assigned_to_host,
                n_host,
                length,
                power_x,
                power_x_t,
                outcome_function,
                std_true_y,
                seed=seed,
            )
        )
        design_data_cand2 = subsample_one_dataset(
            X,
            T,
            p_assigned_to_cand2,
            proportion * length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
            seed=seed,
        )

        data[length] = {
            "host": design_data_host,
            "complementary": design_data_complementary,
            "cand2": design_data_cand2,
        }

    return data


def generate_exact_data_varying_sample_size(
    data_parameters, x_distributions=None, X=None, T=None
):

    n_both_candidates_list, p_assigned_to_host = (
        data_parameters["n_both_candidates_list"],
        data_parameters["p_assigned_to_host"],
    )
    n_host, power_x, power_x_t, outcome_function, std_true_y = (
        data_parameters["n_host"],
        data_parameters["power_x"],
        data_parameters["power_x_t"],
        data_parameters["outcome_function"],
        data_parameters["std_true_y"],
    )

    data = {}

    for seed, length in enumerate(n_both_candidates_list):

        if X is None:  # synthetic
            X, T = generate_rct(x_distributions, seed=seed)
        design_data_host, design_data_complementary = (
            subsample_two_complementary_datasets(
                X,
                T,
                p_assigned_to_host,
                n_host,
                length,
                power_x,
                power_x_t,
                outcome_function,
                std_true_y,
                seed=seed,
            )
        )
        number_x_features = 1 + np.shape(X)[1]
        X_host = design_data_host.iloc[:, :number_x_features]

        # exact_complementary
        complementary_treat = pd.DataFrame(
            [1 if bit == 0 else 0 for bit in design_data_host["T"]], columns=["T"]
        )
        data_complementary = pd.concat(
            [X_host.iloc[:, 1:], complementary_treat], axis=1
        )
        design_data_exact_complementary = generate_design_matrix(
            data_complementary, power_x, power_x_t
        )
        design_data_exact_complementary = add_outcome(
            design_data_exact_complementary, outcome_function, std_true_y
        )

        # exact_twin
        design_data_exact_twin = design_data_host.copy()

        # exact_twin_untreated
        untreated = pd.DataFrame([0] * len(complementary_treat), columns=["T"])
        data_exact_twin_untreated = pd.concat([X_host.iloc[:, 1:], untreated], axis=1)
        design_data_exact_twin_untreated = generate_design_matrix(
            data_exact_twin_untreated, power_x, power_x_t
        )
        design_data_exact_twin_untreated = add_outcome(
            design_data_exact_twin_untreated, outcome_function, std_true_y
        )

        # exact_twin_treated
        treated = pd.DataFrame([1] * len(complementary_treat), columns=["T"])
        data_exact_twin_treated = pd.concat([X_host.iloc[:, 1:], treated], axis=1)
        design_data_exact_twin_treated = generate_design_matrix(
            data_exact_twin_treated, power_x, power_x_t
        )
        design_data_exact_twin_treated = add_outcome(
            design_data_exact_twin_treated, outcome_function, std_true_y
        )

        ### if needed, expansion

        num_samples_needed = length - len(X_host)
        if num_samples_needed > 0:

            sampled_data_complementary = design_data_exact_complementary.sample(
                n=num_samples_needed, replace=True, random_state=0
            )
            design_data_exact_complementary = pd.concat(
                [design_data_exact_complementary, sampled_data_complementary],
                ignore_index=True,
            )

            sampled_data_twin = design_data_exact_twin.sample(
                n=num_samples_needed, replace=True, random_state=0
            )
            design_data_exact_twin = pd.concat(
                [design_data_exact_twin, sampled_data_twin], ignore_index=True
            )

            sampled_data_exact_twin_untreated = design_data_exact_twin_untreated.sample(
                n=num_samples_needed, replace=True, random_state=0
            )
            design_data_exact_twin_untreated = pd.concat(
                [design_data_exact_twin_untreated, sampled_data_exact_twin_untreated],
                ignore_index=True,
            )

            sampled_data_exact_twin_treated = design_data_exact_twin_treated.sample(
                n=num_samples_needed, replace=True, random_state=0
            )
            design_data_exact_twin_treated = pd.concat(
                [design_data_exact_twin_treated, sampled_data_exact_twin_treated],
                ignore_index=True,
            )

        data[length] = {
            "host": design_data_host,
            "exact_complementary": design_data_exact_complementary,
            "exact_twin": design_data_exact_twin,
            "exact_twin_untreated": design_data_exact_twin_untreated,
            "exact_twin_treated": design_data_exact_twin_treated,
        }

    return data
