from typing import Any, Callable
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
import sys


def generate_rct(
    x_sampled_covariates: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame.from_dict(x_sampled_covariates)
    n_global = X.shape[0]
    T = np.random.randint(0, 2, size=n_global)  # Generate T
    return X, T


# Probability functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# Function to generate host and mirror data
def generate_host_and_mirror(
    X: pd.DataFrame,
    T: np.ndarray,  # combine X and T in generate_rct?
    f_assigned_to_host: Callable,  # ??
    n_host: int,
    n_mirror: int,
    power_x: int,
    power_x_t: int,
    outcome_function: Callable,
    std_true_y: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if n_host + n_mirror > X.shape[0]:
        raise ValueError("n_host + n_mirror > n_rct")

    XandT = pd.concat([X, pd.DataFrame(T, columns=["T"])], axis=1)
    # Initialize dataframes for the host and mirror
    data_host = pd.DataFrame(index=range(n_host), columns=XandT.columns, dtype=float)
    data_mirror = pd.DataFrame(
        index=range(n_mirror), columns=XandT.columns, dtype=float
    )

    count_mirror, count_host = 0, 0
    # iterate over rows of X and T
    for _, x_and_t in XandT.iterrows():
        proba_assigned_to_host = f_assigned_to_host(
            x_and_t.drop("T"), x_and_t["T"], np.random.normal()
        )
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)  # 0 or 1

        if is_assigned_to_host and count_host < n_host:
            data_host.loc[count_host] = x_and_t
            count_host += 1
        elif not is_assigned_to_host and count_mirror < n_mirror:
            data_mirror.loc[count_mirror] = x_and_t
            count_mirror += 1

        if count_mirror == n_mirror and count_host == n_host:
            break

    assert (
        len(data_host) == n_host
    ), f"Expected len(data_host) to be {n_host}, got {len(data_host)}"
    assert (
        len(data_mirror) == n_mirror
    ), f"Expected len(data_mirror) to be {n_mirror}, got {len(data_mirror)}"

    design_data_host = generate_design_matrix(data_host, power_x, power_x_t)
    design_data_mirror = generate_design_matrix(data_mirror, power_x, power_x_t)

    design_data_host = add_outcome(design_data_host, outcome_function, std_true_y)
    design_data_mirror = add_outcome(design_data_mirror, outcome_function, std_true_y)

    return design_data_host, design_data_mirror


# Function to generate host2 data
def generate_cand2(
    X, T, f_assigned_to_cand2, n_cand2, power_x, power_x_t, outcome_function, std_true_y
):

    n_global = np.shape(X)[0]
    # Initialize dictionaries
    data_cand2 = {}
    for name in X.columns:
        data_cand2[name] = []

    # Add 'T' key to each dictionary
    data_cand2["T"] = []

    if n_cand2 > n_global:
        print("n_cand2 > n_rct")
        return

    for i in range(n_global):
        proba_assigned_to_cand2 = f_assigned_to_cand2(
            X.iloc[i, :], T[i], np.random.normal()
        )
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 == 1:
            first_value = next(iter(data_cand2.values()))
            if len(first_value) < n_cand2:
                for column_name in X.columns:
                    data_cand2[column_name].append(X.iloc[i][column_name])
                data_cand2["T"].append(T[i])
            else:
                break

    data_cand2 = pd.DataFrame.from_dict(data_cand2)

    if len(data_cand2) != n_cand2:
        print(
            "len(data_cand2) n="
            + str(len(data_cand2))
            + " != n_mirror ("
            + str(n_cand2)
            + ")"
        )

    design_cand2 = generate_design_matrix(data_cand2, power_x, power_x_t)
    design_cand2 = add_outcome(design_cand2, outcome_function, std_true_y)

    return design_cand2


def generate_design_matrix(data, power_x: int, power_x_t: int):
    # Extract X and T from the dataframe
    X = data.drop(columns=["T"])
    T = data["T"]
    n, d = np.shape(X)

    # Initialize the design matrix
    X_prime = pd.DataFrame(index=range(n))
    X_prime["intercept"] = 1.0
    # append X to X_prime
    X_prime = pd.concat([X_prime, X], axis=1)

    # Concatenate X^i for i=2 upto power_x
    for i in range(2, power_x + 1):
        for col in X.columns:
            X_prime[f"{col}**{i}"] = X[col] ** i

    # Concat T and T*X^i for i=1 upto power_x_t
    X_prime["T"] = T
    for i in range(1, power_x_t + 1):
        for col in X.columns:
            colname = f"T*{col}**{i}" if i > 1 else f"T*{col}"
            X_prime[colname] = T * (X[col] ** i)

    assert X_prime.shape == (n, 1 + d * power_x + 1 + d * power_x_t), "Shape mismatch"

    return X_prime


def add_outcome(data, outcome_function, noise_scale):
    n = np.shape(data)[0]
    X = data.drop(columns=["T"])
    T = data["T"]
    eps = np.random.normal(size=n, scale=noise_scale)

    Y = outcome_function(X, T, eps)
    data["Y"] = Y

    return data


def generate_synthetic_data_varying_sample_size(data_parameters, print=True):

    (
        n_both_candidates_list,
        proportion,
        n_rct_before_split,
        x_distributions,
        p_assigned_to_cand2,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
        data_parameters["n_rct_before_split"],
        data_parameters["x_distributions"],
        data_parameters["p_assigned_to_cand2"],
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

    for length in n_both_candidates_list:

        X_rct, T_rct = generate_rct(x_distributions)
        design_data_host, design_data_mirror = generate_host_and_mirror(
            X=X_rct,
            T=T_rct,
            f_assigned_to_host=p_assigned_to_cand2,
            n_host=n_host,
            n_mirror=length,
            power_x=power_x,
            power_x_t=power_x_t,
            outcome_function=outcome_function,
            std_true_y=std_true_y,
        )

        pre_X_cand2, pre_T_cand2 = generate_rct(x_distributions)
        design_data_cand2 = generate_cand2(
            pre_X_cand2,
            pre_T_cand2,
            p_assigned_to_cand2,
            proportion * length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
        )

        data[length] = {
            "host": design_data_host,
            "mirror": design_data_mirror,
            "cand2": design_data_cand2,
        }

    return data


def generate_exact_synthetic_data_varying_sample_size(
    data_parameters: dict[str, Any]
) -> dict[int, dict]:

    (
        n_both_candidates_list,
        proportion,
        n_rct_before_split,
        x_distributions,
        p_assigned_to_cand2,
    ) = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
        data_parameters["n_rct_before_split"],
        data_parameters["x_distributions"],
        data_parameters["p_assigned_to_cand2"],
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

    for length in n_both_candidates_list:
        X_rct, T_rct = generate_rct(x_distributions)  # OK
        design_data_host, _ = generate_host_and_mirror(
            X_rct,
            T_rct,
            p_assigned_to_cand2,
            n_host,
            length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
        )  # mirror isn't used.
        # get the covariates only (no intercept)
        X_host = design_data_host[X_rct.columns]

        # create the exact complementary treatment:
        data_complementary = X_host.copy()
        data_complementary["T"] = 1.0 - design_data_host["T"]
        design_data_exact_complementary = generate_design_matrix(
            data_complementary, power_x, power_x_t
        )
        design_data_exact_complementary = add_outcome(
            design_data_exact_complementary, outcome_function, std_true_y
        )

        # exact_twin
        design_data_exact_twin = design_data_host.copy()

        # exact_twin_untreated: Same covariates but no treatment
        data_exact_twin_untreated = X_host.copy()
        data_exact_twin_untreated["T"] = 0.0
        design_data_exact_twin_untreated = generate_design_matrix(
            data_exact_twin_untreated, power_x, power_x_t
        )
        design_data_exact_twin_untreated = add_outcome(
            design_data_exact_twin_untreated, outcome_function, std_true_y
        )

        # exact_twin_treated:  Same covariates but all treatment
        data_exact_twin_treated = X_host.copy()
        data_exact_twin_treated["T"] = 1.0
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
                n=num_samples_needed, replace=True
            )
            design_data_exact_complementary = pd.concat(
                [design_data_exact_complementary, sampled_data_complementary],
                ignore_index=True,
            )

            sampled_data_twin = design_data_exact_twin.sample(
                n=num_samples_needed, replace=True
            )
            design_data_exact_twin = pd.concat(
                [design_data_exact_twin, sampled_data_twin], ignore_index=True
            )

            sampled_data_exact_twin_untreated = design_data_exact_twin_untreated.sample(
                n=num_samples_needed, replace=True
            )
            design_data_exact_twin_untreated = pd.concat(
                [design_data_exact_twin_untreated, sampled_data_exact_twin_untreated],
                ignore_index=True,
            )

            sampled_data_exact_twin_treated = design_data_exact_twin_treated.sample(
                n=num_samples_needed, replace=True
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


def generate_data_from_real_varying_sample_size(X, T, data_parameters):
    n_both_candidates_list, proportion, p_assigned_to_cand2 = (
        data_parameters["n_both_candidates_list"],
        data_parameters["proportion"],
        data_parameters["p_assigned_to_cand2"],
    )
    n_host, power_x, power_x_t, outcome_function, std_true_y = (
        data_parameters["n_host"],
        data_parameters["power_x"],
        data_parameters["power_x_t"],
        data_parameters["outcome_function"],
        data_parameters["std_true_y"],
    )

    data = {}

    for length in n_both_candidates_list:

        design_data_host, design_data_mirror = generate_host_and_mirror(
            X,
            T,
            p_assigned_to_cand2,
            n_host,
            length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
        )

        design_data_cand2 = generate_cand2(
            X,
            T,
            p_assigned_to_cand2,
            proportion * length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
        )

        data[length] = {
            "host": design_data_host,
            "mirror": design_data_mirror,
            "cand2": design_data_cand2,
        }

    return data


def generate_exact_real_data_varying_sample_size(X, T, data_parameters):

    n_both_candidates_list, p_assigned_to_cand2 = (
        data_parameters["n_both_candidates_list"],
        data_parameters["p_assigned_to_cand2"],
    )
    n_host, power_x, power_x_t, outcome_function, std_true_y = (
        data_parameters["n_host"],
        data_parameters["power_x"],
        data_parameters["power_x_t"],
        data_parameters["outcome_function"],
        data_parameters["std_true_y"],
    )

    data = {}

    for length in n_both_candidates_list:

        design_data_host, design_data_mirror = generate_host_and_mirror(
            X,
            T,
            p_assigned_to_cand2,
            n_host,
            length,
            power_x,
            power_x_t,
            outcome_function,
            std_true_y,
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
                n=num_samples_needed, replace=True
            )
            design_data_exact_complementary = pd.concat(
                [design_data_exact_complementary, sampled_data_complementary],
                ignore_index=True,
            )

            sampled_data_twin = design_data_exact_twin.sample(
                n=num_samples_needed, replace=True
            )
            design_data_exact_twin = pd.concat(
                [design_data_exact_twin, sampled_data_twin], ignore_index=True
            )

            sampled_data_exact_twin_untreated = design_data_exact_twin_untreated.sample(
                n=num_samples_needed, replace=True
            )
            design_data_exact_twin_untreated = pd.concat(
                [design_data_exact_twin_untreated, sampled_data_exact_twin_untreated],
                ignore_index=True,
            )

            sampled_data_exact_twin_treated = design_data_exact_twin_treated.sample(
                n=num_samples_needed, replace=True
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


if __name__ == "__main__":
    n_both_candidates_list = [200]  # , 500, 1000
    proportion = 1  # n_cand2 = prorportion * n_both_candidates_list
    std_true_y = 1
    # set seed
    np.random.seed(42)

    n_rct_before_split = 10**5
    n_host = 200
    sigma_prior = 1
    sigma_rand_error = 1

    power_x, power_x_t = 1, 1
    causal_param_first_index = 4
    outcome_function = (
        # y = 1 + 1*X_0 - 1*X_1 + 1*X_2 + 4*T + 2*X_0*T + 2*X_1*T + 0*X_2*T + eps
        lambda X, T, eps: 1  # intercept, non-causal
        + 1 * X["X_0"]  # non-causal
        - 1 * X["X_1"]  # non-causal
        + 1 * X["X_2"]  # non-causal
        + 4 * T  # causal
        + 2 * X["X_0"] * T  # causal
        + 2 * X["X_1"] * T  # causal
        + 0 * X["X_2"] * T  # causal
        + eps
    )
    true_params = np.array([1, 1, -1, 1, 4, 2, 2, 0])  # copied from above
    std_true_y = 1  # Standard deviation for the true Y

    X0 = np.random.beta(12, 3, size=n_rct_before_split)
    X1 = np.random.normal(loc=4, scale=1, size=n_rct_before_split)
    X2 = np.random.beta(1, 7, size=n_rct_before_split)
    x_distributions = {"X_0": X0, "X_1": X1, "X_2": X2}
    d = 1 + len(x_distributions) * (power_x) + 1 + len(x_distributions) * (power_x_t)

    p_assigned_to_host = lambda X, T, eps: sigmoid(
        1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
    )
    p_assigned_to_cand2 = lambda X, T, eps: sigmoid(
        1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
    )
    # p_assigned_to_cand2 = lambda X_0, X_1, T, eps: sigmoid(1 - 2*X_0 + eps)

    data_parameters = {
        "n_both_candidates_list": n_both_candidates_list,
        "proportion": proportion,
        "n_rct_before_split": n_rct_before_split,
        "x_distributions": x_distributions,
        "p_assigned_to_cand2": p_assigned_to_cand2,
        "n_host": n_host,
        "power_x": power_x,
        "power_x_t": power_x_t,
        "outcome_function": outcome_function,
        "std_true_y": std_true_y,
        "causal_param_first_index": causal_param_first_index,
    }
    data = generate_exact_synthetic_data_varying_sample_size(data_parameters)
    print(data)
