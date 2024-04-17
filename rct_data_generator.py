from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Union


def get_data(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the data for the specified dataset.

    Args:
        dataset: name of the dataset load.
            Currently, the options are 'twins' and 'acic'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: data, covariates, treatment assignment, and outcomes
    """
    if dataset == "twins":
        data = pd.read_csv("data/twins_ztwins_sample0.csv")
        x = data.drop(columns=["y0", "y1", "ite", "y", "t"])
        t = data["t"]
        y = data["y"]

    elif dataset == "acic":
        data = pd.read_csv("./acic_zymu_174570858.csv")
        x = pd.read_csv("./data/acic_x.csv")
        t = data["z"]
        y = data["y0"]
        idx_to_change = data.loc[data["z"] == 1].index.to_list()
        for idx in idx_to_change:
            y.loc[idx] = data["y1"].loc[idx]
        y = y.rename("y")
        one_hot = OneHotEncoder(drop="first").fit(x[["x_2", "x_21", "x_24"]])
        new_data = pd.DataFrame(
            one_hot.transform(x[["x_2", "x_21", "x_24"]]).toarray(),  # type: ignore
        )
        x = x.drop(columns=["x_2", "x_21", "x_24"])
        x = pd.concat([x, new_data], axis=1)
    else:
        raise ValueError(f"Dataset {dataset} not recognized")

    return data, x, t, y


def generate_rct(
    x_sampled_covariates: dict[str, np.ndarray], seed: Union [int, None] = 0
) -> pd.DataFrame:
    """
    Generate a randomised controlled trial (RCT) dataset.

    Args:
        x_sampled_covariates: dictionary of covariates with keys as column names

    Returns:
        pd.DataFrame: dataframe containing covariates and a column "T" corresponding to
            treatment assignment
    """
    if seed is not None:
        np.random.seed(seed)
    X = pd.DataFrame.from_dict(x_sampled_covariates)
    X["T"] = np.random.randint(0, 2, size=X.shape[0])  # Generate T
    return X


def generate_design_matrix(
    data: pd.DataFrame,
    power_x: int = 1,
    power_x_t: int = 1,
    include_intercept: bool = True,
) -> pd.DataFrame:
    """
    Generate the design matrix from `data` with polynomial features of
        `power_x` and `power_x_t`.

    Args:
        data: dataframe containing covariates and a column "T" corresponding to
            treatment assignment
        power_x: if > 1, add polynomial features of the covariates up to this power
        power_x_t: if > 1, add polynomial features of the covariates times treatment up
            to this power

    Returns:
        pd.DataFrame: design matrix with a total of
            1 + d * power_x + 1 + d * power_x_t columns (i.e. includes intercept)
    """
    X = data.drop(columns=["T"])
    T = data["T"]
    n, d = np.shape(X)

    # Initialize the design matrix
    X_prime = pd.DataFrame(index=range(n))
    if include_intercept:
        X_prime["intercept"] = 1.0  # interept column, set to 1
    X_prime = pd.concat([X_prime, X], axis=1)  # append X to X_prime

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

    assert X_prime.shape == (
        n,
        include_intercept + d * power_x + 1 + d * power_x_t,
    ), "Shape mismatch"

    return X_prime


def append_outcome(
    data: pd.DataFrame,
    outcome_function: Callable,
    noise_scale: Union [float, None] = None,
    eps: Union [np.ndarray, None] = None,
) -> pd.DataFrame:
    """
    Generate y = outcome_function(X, T) + N(0, noise_scale)
        and append the outcome `y` to `data`.

    Args:
        data: dataframe containing covariates and a column "T" corresponding to
            treatment assignment
        outcome_function: function to generate the outcome
        noise_scale: standard deviation of the noise to add to the outcome

    Returns:
        A dataframe with the outcome appended
    """
    if eps is None:
        assert noise_scale is not None, "Need noise_scale if eps is None"
        eps = np.random.normal(size=len(data), scale=noise_scale)
    data["Y"] = outcome_function(data.drop(columns=["T"]), data["T"], eps)
    return data


def sigmoid(x: np.ndarray) -> np.ndarray:
    # helper
    return 1.0 / (1.0 + np.exp(-x))


def subsample_two_complementary_datasets(
    XandT: pd.DataFrame,
    f_assigned_to_host: Callable,  # ??
    n_host: int,
    n_complementary: int,
    power_x: int,
    power_x_t: int,
    outcome_function: Callable,
    std_true_y: float,
    include_intercept: bool = True,
    seed: Union[int, None] = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate host and its complementary data for a synthetic RCT.

    Args:
        XandT: dataframe containing covariates and a column "T" corresponding to
            treatment assignment
        n_complementary: number of samples for the "complementary" set (best we can get)
        power_x: power of the covariates
        power_x_t: power of the covariates times treatment
        outcome_function: function to generate the outcome
        std_true_y: standard deviation of the true outcome
        include_intercept: whether to include an intercept in the design matrix
        seed: random seed

    Raises:
        ValueError: if n_host + n_complementary > X.shape[0]

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: host and complementary datasets
    """

    if n_host + n_complementary > XandT.shape[0]:
        raise ValueError("n_host + n_complementary > n_rct")
    if seed is not None:
        np.random.seed(seed)

    # Initialize dataframes for the host and its complementary
    data_host = pd.DataFrame(index=range(n_host), columns=XandT.columns, dtype=float)
    data_complementary = pd.DataFrame(
        index=range(n_complementary), columns=XandT.columns, dtype=float
    )

    count_complementary, count_host = 0, 0
    for _, x_and_t in XandT.iterrows():
        proba_assigned_to_host = f_assigned_to_host(
            x_and_t.drop("T"), x_and_t["T"], np.random.normal()
        )
        is_assigned_to_host = np.random.binomial(1, proba_assigned_to_host)  # 0 or 1

        if is_assigned_to_host and count_host < n_host:
            data_host.loc[count_host] = x_and_t
            count_host += 1
        elif not is_assigned_to_host and count_complementary < n_complementary:
            data_complementary.loc[count_complementary] = x_and_t
            count_complementary += 1

        if count_complementary == n_complementary and count_host == n_host:
            break

    assert (
        len(data_host) == n_host
    ), f"Expected len(data_host) to be {n_host}, got {len(data_host)}"
    assert (
        len(data_complementary) == n_complementary
    ), f"Expected len(n_complementary)={n_complementary}, got {len(data_complementary)}"

    design_data_host = generate_design_matrix(
        data_host, power_x, power_x_t, include_intercept
    )
    design_data_complementary = generate_design_matrix(
        data_complementary, power_x, power_x_t, include_intercept
    )

    design_data_host = append_outcome(design_data_host, outcome_function, std_true_y)
    design_data_complementary = append_outcome(
        design_data_complementary, outcome_function, std_true_y
    )

    return design_data_host, design_data_complementary


# Function to generate the second candidate dataset
def subsample_one_dataset(
    XandT: pd.DataFrame,
    assignment_function: Callable,
    sample_size: int,
    power_x: int,
    power_x_t: int,
    outcome_function: Callable,
    std_true_y: float,
    include_intercept: bool = True,
    seed: Union [int, None] = 0,
) -> pd.DataFrame:
    if sample_size > XandT.shape[0]:
        raise ValueError("sample_size > n_global")
    if seed is not None:
        np.random.seed(seed)

    data = pd.DataFrame(index=range(sample_size), columns=XandT.columns, dtype=float)
    count_cand2 = 0
    for _, x_and_t in XandT.iterrows():
        proba_assigned_to_cand2 = assignment_function(
            x_and_t.drop("T"), x_and_t["T"], np.random.normal()
        )
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 and count_cand2 < sample_size:
            data.loc[count_cand2] = x_and_t
            count_cand2 += 1

        if count_cand2 == sample_size:
            break

    assert (
        len(data) == sample_size
    ), f"Expected len(data)={sample_size}, got {len(data)}"
    design_data = generate_design_matrix(data, power_x, power_x_t, include_intercept)
    design_data = append_outcome(design_data, outcome_function, std_true_y)
    return design_data


def generate_data_varying_sample_size(
    data_parameters: dict[str, Any],
    X_rct: Union [pd.DataFrame, None] = None,
    T_rct: Union [np.ndarray, None] = None,
    include_intercept: bool = True,
) -> dict[int, dict]:
    # if X_rct and T_rct are None, generate them; if not, use them
    # need "x_distributions" to bein the data_parameters if X_rct and T_rct are None
    x_distributions = data_parameters.get("x_distributions", None)
    power_x = data_parameters["power_x"]
    power_x_t = data_parameters["power_x_t"]
    outcome_function = data_parameters["outcome_function"]
    std_true_y = data_parameters["std_true_y"]

    data = {}

    for seed, length in enumerate(data_parameters["n_both_candidates_list"]):
        # should the generation be outside of the loop actually?
        if X_rct is None and T_rct is None:
            assert (
                x_distributions is not None
            ), "Need x_distributions if X_rct and T_rct are None"
            XandT = generate_rct(x_distributions, seed=seed)
        else:
            assert X_rct is not None and T_rct is not None, "Need both X_rct and T_rct"
            XandT = pd.concat([X_rct, pd.DataFrame(T_rct, columns=["T"])], axis=1)

        design_data_host, design_data_comp = subsample_two_complementary_datasets(
            XandT=XandT,
            f_assigned_to_host=data_parameters["p_assigned_to_host"],  # host?
            n_host=data_parameters["n_host"],
            n_complementary=length,
            power_x=power_x,
            power_x_t=power_x_t,
            outcome_function=outcome_function,
            std_true_y=std_true_y,
            include_intercept=include_intercept,
            seed=seed,
        )

        design_data_cand2 = subsample_one_dataset(
            XandT=XandT,
            assignment_function=data_parameters["p_assigned_to_cand2"],
            sample_size=data_parameters["proportion"] * length,
            power_x=power_x,
            power_x_t=power_x_t,
            outcome_function=outcome_function,
            std_true_y=std_true_y,
            include_intercept=include_intercept,
            seed=seed,
        )

        data[length] = {
            "host": design_data_host,
            "complementary": design_data_comp,
            "cand2": design_data_cand2,
        }

    return data


def generate_exact_data_varying_sample_size(
    data_parameters: dict[str, Any],
    X_rct: Union [pd.DataFrame, None] = None,
    T_rct: Union [np.ndarray, None] = None,
    include_intercept: bool = True,
) -> dict[int, dict]:
    n_host = data_parameters["n_host"]
    power_x = data_parameters["power_x"]
    power_x_t = data_parameters["power_x_t"]
    outcome_function = data_parameters["outcome_function"]
    std_true_y = data_parameters["std_true_y"]

    data = {}

    for seed, length in enumerate(data_parameters["n_both_candidates_list"]):
        # should the generation be outside of the loop actually?
        if X_rct is None and T_rct is None:
            XandT = generate_rct(data_parameters["x_distributions"], seed=seed)
        else:
            assert X_rct is not None and T_rct is not None, "Need both X_rct and T_rct"
            XandT = pd.concat([X_rct, pd.DataFrame(T_rct, columns=["T"])], axis=1)

        design_data_host, _ = subsample_two_complementary_datasets(
            XandT=XandT,
            f_assigned_to_host=data_parameters["p_assigned_to_host"],
            n_host=n_host,
            n_complementary=length,
            power_x=power_x,
            power_x_t=power_x_t,
            outcome_function=outcome_function,
            std_true_y=std_true_y,
            include_intercept=include_intercept,
            seed=seed,
        )  # complementary data isn't used.

        # get the covariates only (no intercept)
        X_host = design_data_host[XandT.columns]
        assert n_host == len(X_host), "Shape mismatch"

        # create the exact complementary treatment:
        data_complementary = X_host.copy()
        data_complementary["T"] = 1.0 - design_data_host["T"]  # overwrite T with 1 - T
        design_data_exact_complementary = generate_design_matrix(
            data_complementary, power_x, power_x_t, include_intercept
        )
        design_data_exact_complementary = append_outcome(
            design_data_exact_complementary, outcome_function, std_true_y
        )

        # exact_twin
        design_data_exact_twin = design_data_host.copy()

        # exact_twin_untreated: Same covariates but no treatment
        data_exact_twin_untreated = X_host.copy()
        data_exact_twin_untreated["T"] = 0.0  # overwrite T with all 0s
        design_data_exact_twin_untreated = generate_design_matrix(
            data_exact_twin_untreated, power_x, power_x_t, include_intercept
        )
        design_data_exact_twin_untreated = append_outcome(
            design_data_exact_twin_untreated, outcome_function, std_true_y
        )

        # exact_twin_treated:  Same covariates but all treatment
        data_exact_twin_treated = X_host.copy()
        data_exact_twin_treated["T"] = 1.0  # overwrite T with all 1s
        design_data_exact_twin_treated = generate_design_matrix(
            data_exact_twin_treated, power_x, power_x_t, include_intercept
        )
        design_data_exact_twin_treated = append_outcome(
            design_data_exact_twin_treated, outcome_function, std_true_y
        )
        data[length] = {
            "host": design_data_host,
            "exact_complementary": design_data_exact_complementary,
            "exact_twin": design_data_exact_twin,
            "exact_twin_untreated": design_data_exact_twin_untreated,
            "exact_twin_treated": design_data_exact_twin_treated,
        }
        # if more samples are needed, subsample from the existing data
        if length > n_host:
            for name, df in data[length].items():
                sampled_data = df.sample(n=length - n_host, replace=True)
                data[length][name] = pd.concat([df, sampled_data], ignore_index=True)

    return data


def _main():
    n_both_candidates_list = [200]  # , 500, 1000
    proportion = 1  # n_cand2 = prorportion * n_both_candidates_list
    std_true_y = 1
    np.random.seed(42)

    n_rct_before_split = 10**5
    n_host = 200

    power_x, power_x_t = 1, 1
    causal_param_first_index = 4
    outcome_function = (
        # y = 1 + 1*X_0 - 1*X_1 + 1*X_2 + 4*T + 2*X_0*T + 2*X_1*T + 0*X_2*T + eps
        lambda X, T, eps: 0  # intercept, non-causal; 0 => no intercept
        + 1 * X["X_0"]  # non-causal
        - 1 * X["X_1"]  # non-causal
        + 1 * X["X_2"]  # non-causal
        + 4 * T  # causal
        + 2 * X["X_0"] * T  # causal
        + 2 * X["X_1"] * T  # causal
        + 0 * X["X_2"] * T  # causal
        + eps
    )
    true_params = np.array([1, -1, 1, 4, 2, 2, 0])  # copied from above
    std_true_y = 1  # Standard deviation for the true Y

    X0 = np.random.beta(12, 3, size=n_rct_before_split)
    X1 = np.random.normal(loc=4, scale=1, size=n_rct_before_split)
    X2 = np.random.beta(1, 7, size=n_rct_before_split)
    x_distributions = {"X_0": X0, "X_1": X1, "X_2": X2}
    d = 1 + len(x_distributions) * (power_x) + 1 + len(x_distributions) * (power_x_t)

    p_assigned_to_host = lambda X, T, eps: sigmoid(
        1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
    )  # not used
    p_assigned_to_cand2 = lambda X, T, eps: sigmoid(
        1 + 2 * X["X_0"] - X["X_1"] + 2 * T + eps
    )
    # p_assigned_to_cand2 = lambda X_0, X_1, T, eps: sigmoid(1 - 2*X_0 + eps)

    # doesn't have p_assigned_to_host!
    data_parameters = {
        "n_both_candidates_list": n_both_candidates_list,
        "proportion": proportion,
        "n_rct_before_split": n_rct_before_split,
        "x_distributions": x_distributions,
        "p_assigned_to_cand2": p_assigned_to_cand2,
        "p_assigned_to_host": p_assigned_to_host,
        "n_host": n_host,
        "power_x": power_x,
        "power_x_t": power_x_t,
        "outcome_function": outcome_function,
        "std_true_y": std_true_y,
        "causal_param_first_index": causal_param_first_index,
    }
    data = generate_exact_data_varying_sample_size(
        data_parameters, include_intercept=False
    )
    return data, data_parameters


if __name__ == "__main__":
    data = _main()
    print(data)
