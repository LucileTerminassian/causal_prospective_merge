from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Union, List
from causallib import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_data(dataset: str, path: str, th = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get the data for the specified dataset.

    Args:
        dataset: name of the dataset load.
            Currently, the options are 'twins' and 'acic'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: data, covariates, treatment assignment, and outcomes
    """

    if dataset == 'acic':
        data = datasets.data_loader.load_acic16()
        del data['descriptors'] 
        ground_truth = data['po'].rename(columns={'0': 'y0', '1': 'y1'})
        x = data['X']

        # selected_columns = x.filter(regex='^x_(2|21|24)')
        # one_hot = OneHotEncoder(drop="first").fit(selected_columns)
        # new_data = pd.DataFrame(
        #     one_hot.transform(selected_columns).toarray(),  # type: ignore
        # )
        columns_to_drop = x.filter(regex='^x_(2|21|24)').columns
        x = x.drop(columns=columns_to_drop)
        
        # x = pd.concat([x, new_data], axis=1)
        data = pd.concat([x, data['a'], data['y'], ground_truth], axis=1)
        data.dropna(inplace=True)
        data.rename(columns={'a': 'T', 0: 'Y'}, inplace=True)
        x = data.drop(columns=["y0", "y1", "Y", "T"])
        t = data['T']
        y = data['Y']

    elif dataset == "twins":
        data = pd.read_csv(path + "data/twins_ztwins_sample0.csv")
        data.dropna(inplace=True)
        data.rename(columns = {'t': 'T', 'y': 'Y'}, inplace=True)
        x = data.drop(columns=["y0", "y1", "ite", "Y", "T"])
        t = data["T"]
        y = data["Y"]
    
    elif dataset == "IDHP":
        data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
        col =  ["T", "Y", "y_cf", "y0", "y1" ,]
        for i in range(1,26):
            col.append("x"+str(i))
        data.columns = col
        data = data.astype({"T":'float'}, copy=False)
        data = data.drop(columns=["y_cf"])
        x = data[["x"+str(i) for  i in range(1,6)]]
        IDHP_scalar = StandardScaler().fit(x)
        x = IDHP_scalar.transform(x)
        data[["x"+str(i) for  i in range(1,6)]] = x
        x = data[["x"+str(i) for  i in range(1,26)]]
        t = data['T']
        y = data['Y']

    elif dataset == 'lalonde':
        data = pd.read_csv( path+ "data/lalonde_cps_sample0.csv")
        data.dropna(inplace=True)
        data.rename(columns = {'t': 'T', 'y': 'Y'}, inplace=True)
        if th is not None:
            x = data.drop(columns=["y0", "y1", "ite", "Y", "T"])
            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(x, data['T'])
            propensity_scores = logistic_model.predict_proba(x)[:, 1]
            data.loc[propensity_scores > th, 'T'] = 1
            print('portion of treated is '+str(np.sum(data['T'])/len(data['T'])))
        data['Y'] = data['y1'] * data['T'] + data['y0'] * (1 - data['T'])
        x = data.drop(columns=["y0", "y1", "ite", "Y", "T"])
        t = data["T"]
        y = data["Y"]

    else:
        raise ValueError(f"Dataset {dataset} not recognized")
    
    y_std = y.std()
    y_mean = y.mean()
    y = (y - y.mean())/y_std
    data['Y'] = y

    if "y0" in data.columns:
        data[["y0","y1"]] = (data[["y0","y1"]] - y_mean) /y_std

    return data, x, t, y


def generating_random_sites_from(XandT, data_with_groundtruth, exp_parameters, added_T_coef=1, binary_outcome=False):
    
    candidate_sites = {}
    sample_size, number_features = np.shape(XandT)[0], np.shape(XandT)[1]
    function_indices = {0: lambda X: 1, 1: lambda X: X**3, 2: lambda X: X, 3: lambda X: X**2 }
    number_of_candidate_sites = exp_parameters['number_of_candidate_sites']
    min_sample_size_cand = exp_parameters['min_sample_size_cand']
    max_sample_size_cand = exp_parameters['max_sample_size_cand']
    outcome_function = None
    std_true_y = exp_parameters['std_true_y']
    power_x = exp_parameters['power_x']
    min_treat_group_size = exp_parameters['min_treat_group_size']
    power_x_t = exp_parameters['power_x_t']
    created_sites = 0
    coef_sample_width = exp_parameters['coef_sample_width']

    while created_sites < number_of_candidate_sites : # inforce + 1 cause we also subsample a host site

        # np.random.seed(np.random.randint(10000))
        
        selected_features_for_subsampling = np.random.randint(2, size = number_features) 
        # binary bool vector representing selection for being an input of the sampling function

        if created_sites==0:
            random_coefs = [np.random.uniform(-coef_sample_width/2, coef_sample_width/2) for _ in range(number_features)] 
        else:
            random_coefs = [np.random.uniform(-coef_sample_width/2, coef_sample_width/2) for _ in range(number_features)] 
            
        random_fct_idx = [np.random.randint(0, len(function_indices.keys())) for _ in range(number_features)] 
        
        def p_assigned_to_site(X, T,eps):
            result = 0
            for j in range(number_features-1):
                result += selected_features_for_subsampling[j] * random_coefs[j] * function_indices[random_fct_idx[j]](X[j])
            # here i use added_T_coef * random_coefs to increase importance of T
            result +=  added_T_coef * random_coefs[-1] *  function_indices[random_fct_idx[-1]](T) # T always selected in the end
            return sigmoid(result)
        

        if created_sites==0:
            sample_size = exp_parameters['host_sample_size']+ exp_parameters['host_test_size']

        else:
            sample_size = np.random.randint(min_sample_size_cand, max_sample_size_cand + 1)  # Add 1 to include max_sample_size_cand

        design_data_cand = subsample_one_dataset(XandT, p_assigned_to_site, sample_size, power_x, power_x_t, outcome_function, std_true_y, seed=np.random.randint(10000))
        design_data_cand = design_data_cand.dropna()
        any_nan = design_data_cand.isna().any().any()
        at_least_30_treated = np.sum(design_data_cand["T"]) > min_treat_group_size
        at_least_30_untreated = len(design_data_cand["T"])-np.sum(design_data_cand["T"]) > min_treat_group_size
        candidate = pd.concat([design_data_cand, data_with_groundtruth.loc[design_data_cand.index, 'Y']], axis=1)

        if binary_outcome:
            at_least_20_y_equal1 = np.sum(candidate["Y"]) > 20
            at_least_20_y_equal0 = len(candidate["Y"])-np.sum(candidate["Y"]) > 20
        else:
            at_least_20_y_equal1 = at_least_20_y_equal0 = True

        if not design_data_cand.empty and not any_nan and at_least_30_treated and at_least_30_untreated and at_least_20_y_equal1 and at_least_20_y_equal0: 
            # we're appending
            candidate_sites[created_sites] = candidate
            created_sites += 1
        else:
            pass # not appending

            
    return candidate_sites

def generate_rct(
    x_sampled_covariates: dict[str, np.ndarray], seed: int = 0
) -> pd.DataFrame:
    """
    Generate a randomised controlled trial (RCT) dataset.

    Args:
        x_sampled_covariates: dictionary of covariates with keys as column names

    Returns:
        pd.DataFrame: dataframe containing covariates and a column "T" corresponding to
            treatment assignment
    """
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
    X_prime = pd.DataFrame(index=X.index)
    if include_intercept:
        X_prime["intercept"] = 1.0  # intercept column, set to 1
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
    outcome_function, #: Callable
    noise_scale = None, #: Union [float, None] 
    eps = None, #Union [np.ndarray, None]
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
    f_assigned_to_host,  # : Callable
    n_host: int,
    n_complementary: int,
    power_x: int,
    power_x_t: int,
    outcome_function, #: Callable
    std_true_y: float,
    include_intercept: bool = True,
    seed: int = 0,
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
    assignment_function, #: Callable
    sample_size: int,
    power_x: int,
    power_x_t: int,
    outcome_function, #: Callable
    std_true_y: float,
    include_intercept: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    if sample_size > XandT.shape[0]:
        raise ValueError("sample_size > n_global")
    # np.random.seed(seed)


    data = pd.DataFrame(index=range(sample_size), columns=XandT.columns, dtype=float) 
    count_cand2 = 0
    for _, x_and_t in XandT.iterrows():
        proba_assigned_to_cand2 = assignment_function(
            x_and_t.drop("T"), x_and_t["T"], np.random.normal()
        )
        is_assigned_to_cand2 = np.random.binomial(1, proba_assigned_to_cand2)
        if is_assigned_to_cand2 and count_cand2 < sample_size:
            # data.loc[count_cand2] = x_and_t
            data.iloc[count_cand2] = x_and_t
            data = data.rename(index={count_cand2: x_and_t.name})
            count_cand2 += 1

        if count_cand2 == sample_size:
            break

    assert (
        len(data) == sample_size
    ), f"Expected len(data)={sample_size}, got {len(data)}"
    design_data = generate_design_matrix(data, power_x, power_x_t, include_intercept)
    if outcome_function is not None:
        design_data = append_outcome(design_data, outcome_function, std_true_y)
    return design_data


def generate_data_varying_sample_size(
    data_parameters, #: dict[str, Any]
    X_rct = None, #Union [pd.DataFrame, None] 
    T_rct = None, #Union [np.ndarray, None]
    include_intercept: bool = True,
    seed: int=0,
) -> dict[int, dict]:
    # if X_rct and T_rct are None, generate them; if not, use them
    # need "x_distributions" to bein the data_parameters if X_rct and T_rct are None
    x_distributions = data_parameters.get("x_distributions", None)
    power_x = data_parameters["power_x"]
    power_x_t = data_parameters["power_x_t"]
    outcome_function = data_parameters["outcome_function"]
    std_true_y = data_parameters["std_true_y"]

    data = {}
    np.random.seed(seed)
    seed_for_each_length = np.random.randint(0, 1001, size=len(data_parameters["varying_sample_sizes"]))

    for i, length in enumerate(data_parameters["varying_sample_sizes"]):
        # should the generation be outside of the loop actually?
        if X_rct is None and T_rct is None:
            assert (
                x_distributions is not None
            ), "Need x_distributions if X_rct and T_rct are None"
            XandT = generate_rct(x_distributions, seed=seed_for_each_length[i])
        else:
            assert X_rct is not None and T_rct is not None, "Need both X_rct and T_rct"
            XandT = pd.concat([X_rct, pd.DataFrame(T_rct, columns=["T"])], axis=1)

        if data_parameters["fixed_n_complementary"] is not None: # only cand2 length varies
            design_data_host, design_data_comp = subsample_two_complementary_datasets(
                XandT=XandT,
                f_assigned_to_host=data_parameters["p_assigned_to_host"],  # host?
                n_host=data_parameters["n_host"],
                n_complementary=data_parameters["fixed_n_complementary"],
                power_x=power_x,
                power_x_t=power_x_t,
                outcome_function=outcome_function,
                std_true_y=std_true_y,
                include_intercept=include_intercept,
                seed=seed_for_each_length[i],
            )
        else:
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
                seed=seed_for_each_length[i],
            )

        design_data_cand2 = subsample_one_dataset(
            XandT=XandT,
            assignment_function=data_parameters["p_assigned_to_cand2"],
            sample_size=length,
            power_x=power_x,
            power_x_t=power_x_t,
            outcome_function=outcome_function,
            std_true_y=std_true_y,
            include_intercept=include_intercept,
            seed=seed_for_each_length[i],
        )

        data[length] = {
            "host": design_data_host,
            "complementary": design_data_comp,
            "cand2": design_data_cand2,
        }

    return data


def generate_exact_data_varying_sample_size(
    data_parameters, #: dict[str, Any]
    X_rct = None, #: Union [pd.DataFrame, None]
    T_rct = None, # : Union [np.ndarray, None]
    include_intercept: bool = True,
    seed: int=0,
) -> dict[int, dict]:
    n_host = data_parameters["n_host"]
    power_x = data_parameters["power_x"]
    power_x_t = data_parameters["power_x_t"]
    outcome_function = data_parameters["outcome_function"]
    std_true_y = data_parameters["std_true_y"]

    data = {}
    np.random.seed(seed)
    seed_for_each_length = np.random.randint(0, 1001, size=len(data_parameters["varying_sample_sizes"]))

    for i, length in enumerate(data_parameters["varying_sample_sizes"]):
        # should the generation be outside of the loop actually?
        if X_rct is None and T_rct is None:
            XandT = generate_rct(data_parameters["x_distributions"], seed=seed_for_each_length[i])
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
            seed=seed_for_each_length[i],
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
    varying_sample_sizes = [200]  # , 500, 1000
    proportion = 1  # n_cand2 = prorportion * varying_sample_sizes
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
        "varying_sample_sizes": varying_sample_sizes,
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
