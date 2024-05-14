import crypten
import torch
from crypten.config import cfg
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def chol_LD_Crypten(crypt_PSD_mat):

    L_enc = crypten.cryptensor(torch.zeros(crypt_PSD_mat.shape))
    D_enc = crypten.cryptensor(torch.zeros(crypt_PSD_mat.shape[0]))

    for i in (range(crypt_PSD_mat.shape[0])):
        L_enc[i,i] = 1 
        for j in range(crypt_PSD_mat.shape[0]):
            if i == j:
                if i == 0:
                    D_enc[i] = crypt_PSD_mat[i,i]
                else:
                    D_enc[i] = (crypt_PSD_mat[j,j] - L_enc[j,:j].t() @ (L_enc[j,:j]*D_enc[:j]))
            if i > j:
                    if j>0:
                        L_enc[i,j] = (crypt_PSD_mat[i,j] - L_enc[i,:j].t()@ (L_enc[j,:j]*D_enc[:j])).div(D_enc[j])
                    else:
                        L_enc[i,j] = (crypt_PSD_mat[i,j]).div(D_enc[j])
    
    return L_enc, D_enc

def logdet_Crypten(crypt_PSD_mat):
        _,D_enc = chol_LD_Crypten(crypt_PSD_mat)
        D_log_enc = D_enc.log()
        return D_log_enc.sum()

def compare_obs_EIG_lin(X_host,X_cand,cov,DP=False,epsilon=0.01):
        results_dict = {}

        XX_cand = X_cand.T @ X_cand
        X_host_plus_cov = X_host.T @ X_host + cov
        X_host_plus_cov_inv = torch.linalg.inv(X_host_plus_cov)
        _,EIG_torch = torch.slogdet(XX_cand @ X_host_plus_cov_inv + torch.eye(XX_cand.shape[0]))
        results_dict["EIG_obs_torch"] = EIG_torch.item()

        X_host_crypten = crypten.mpc.MPCTensor(X_host_plus_cov_inv)
        XX_cand_crypten = crypten.mpc.MPCTensor(XX_cand)
        EIG_crypten = logdet_Crypten(XX_cand_crypten @ X_host_crypten + torch.eye(XX_cand.shape[0])).get_plain_text()
        results_dict["EIG_obs_crypten"] = EIG_crypten.item()

        if DP:
            EIG_DP = get_diff_private_obs_EIG(X_host,X_cand,np.array(cov),epsilon=epsilon)
            results_dict["EIG_obs_DP"] = EIG_DP

        return results_dict

def compare_caus_EIG_lin(X_host,X_cand,cov,causal_param_first_index,DP=False,epsilon=0.01):
        results_dict = {}
        

        XX_cand = X_cand.T @ X_cand
        X_host_plus_cov = X_host.T @ X_host + cov
        XX_cand_causal,X_host_plus_cov = XX_cand[causal_param_first_index:,causal_param_first_index:],X_host_plus_cov[causal_param_first_index:,causal_param_first_index:]
        X_host_plus_cov_inv = torch.linalg.inv(X_host_plus_cov)
        _,EIG_torch = torch.slogdet(XX_cand_causal @ X_host_plus_cov_inv + torch.eye(X_host_plus_cov_inv.shape[1]))
        results_dict["EIG_caus_torch"] = EIG_torch.item()

        X_host_crypten = crypten.mpc.MPCTensor(X_host_plus_cov_inv)
        XX_cand_crypten = crypten.mpc.MPCTensor(XX_cand_causal)
        EIG_crypten = logdet_Crypten(XX_cand_crypten @ X_host_crypten + torch.eye(X_host_plus_cov_inv.shape[0])).get_plain_text()
        results_dict["EIG_caus_crypten"] = EIG_crypten.item()

        if DP:
            EIG_DP = get_diff_private_caus_EIG(X_host,X_cand,np.array(cov),causal_param_first_index,epsilon=epsilon)
            results_dict["EIG_caus_DP"] = EIG_DP
        
        return results_dict


# Functions for DP Lin Reg below 

def get_sensativitiy(X,y):

    sens_x = np.zeros(X.shape[1])

    for i in range((X.shape[1])):
        sens_x[i] = X[:,i].max() - X[:,i].min()
    
    sens_y = y.max() - y.min()

    return sens_x,sens_y

def symmetrize(B):

    B = B.copy()

    flip = False
    if len(B.shape) == 4:
        d = B.shape[0]
        B = B.reshape((d ** 2, d ** 2))
        flip = True

    B = np.triu(B) + np.triu(B, k=1).T

    if flip:
        B = B.reshape((d, d, d, d))

    return B

def get_diff_private_version(X,y,epsilon=0.01):

    sensitivity_x, sensitivity_y = get_sensativitiy(X,y)
    S = {'XX': X.T.dot(X),
    'Xy': X.T.dot(y),
    'yy': y .dot(y)
    }

    data_dim = S['XX'].shape[0]

    XX_comps = data_dim * (data_dim + 1) / 2  # upper triangular, not counting last column which is X
    X_comps = data_dim  # last column
    Xy_comps = data_dim
    yy_comps = 1
    sensitivity = XX_comps * sum(sensitivity_x[:-1]) ** 2 \
                    + X_comps * sum(sensitivity_x[:-1]) \
                    + Xy_comps * sum(sensitivity_x[:-1]) * sensitivity_y \
                    + yy_comps * sensitivity_y ** 2

    Z = {key: np.random.laplace(loc=val, scale=sensitivity / epsilon) for key, val in S.items()}

    # symmetrize Z_XX since we only want to add noise to upper triangle
    Z['XX'] = symmetrize(Z['XX'])

    Z['X'] = Z['XX'][:, 0][:, None]

    return Z

def get_diff_private_obs_EIG(X_host,X_cand,cov,epsilon=0.01):


    X_host_np = np.array(X_host)
    X_cand_np = np.array(X_cand)

    y = np.zeros(len(X_host_np))

    Z = get_diff_private_version(X_host_np,y,epsilon=epsilon)

    XX_host = Z["XX"]

    # _,logdet_1 = np.linalg.slogdet(X_cand_np.T @ X_cand_np + XX_host + cov)
    # _,logdet_2 = np.linalg.slogdet(XX_host + cov)
    
    _,logdet_EIG = np.linalg.slogdet((X_cand_np.T @ X_cand_np) @ np.linalg.inv(XX_host+ cov) +np.eye(cov.shape[0]))
    return logdet_EIG

def get_diff_private_caus_EIG(X_host,X_cand,cov,causal_param_first_index,epsilon=0.01):


    X_host_np = np.array(X_host)
    X_cand_np = np.array(X_cand)

    zero_mat = np.zeros_like(X_host_np)
    zero_mat[causal_param_first_index:,causal_param_first_index:] = X_host_np[causal_param_first_index:,causal_param_first_index:]

    y = np.zeros(len(X_host_np))

    Z = get_diff_private_version(X_host_np,y,epsilon=epsilon)

    XX_host = Z["XX"][causal_param_first_index:,causal_param_first_index:]
    XX_cand = (X_cand_np.T @ X_cand_np)[causal_param_first_index:,causal_param_first_index:]
    
    # _,logdet_1 = np.linalg.slogdet(XX_cand + XX_host + cov[causal_param_first_index:,causal_param_first_index:])
    # _,logdet_2 = np.linalg.slogdet(XX_host + cov[causal_param_first_index:,causal_param_first_index:])
    
    _,logdet_EIG = np.linalg.slogdet(XX_cand + np.linalg.inv(XX_host + cov[causal_param_first_index:,causal_param_first_index:]) +np.eye(XX_cand.shape[0]))
    return logdet_EIG

# Functions copied below in correct form for python 3.7


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

def sigmoid(x: np.ndarray) -> np.ndarray:
    # helper
    return 1.0 / (1.0 + np.exp(-x))

def generating_random_sites_from(XandT, data_with_groundtruth, exp_parameters, added_T_coef=1, binary_outcome=False):
    
    candidate_sites = {}
    sample_size, number_features = np.shape(XandT)[0], np.shape(XandT)[1]
    function_indices = {0: lambda X: np.tan(X), 1: lambda X: X**3, 2: lambda X: X, 3: lambda X: X**2 }
    number_of_candidate_sites = exp_parameters['number_of_candidate_sites']
    min_sample_size_cand = exp_parameters['min_sample_size_cand']
    max_sample_size_cand = exp_parameters['max_sample_size_cand']
    outcome_function = None
    std_true_y = exp_parameters['std_true_y']
    power_x = exp_parameters['power_x']
    power_x_t = exp_parameters['power_x_t']
    created_sites = 0
    
    while created_sites < number_of_candidate_sites : # inforce + 1 cause we also subsample a host site

        np.random.seed(np.random.randint(10000))
        
        selected_features_for_subsampling = np.random.randint(2, size = number_features) 
        # binary bool vector representing selection for being an input of the sampling function
        random_coefs = [np.random.uniform(-10, 10) for _ in range(number_features)] 
        random_fct_idx = [np.random.randint(0, len(function_indices.keys())) for _ in range(number_features)] 
        
        def p_assigned_to_site(X, T, eps):
            result = 0
            for j in range(number_features-1):
                result += selected_features_for_subsampling[j] * random_coefs[j] * function_indices[random_fct_idx[j]](X[j])
            # here i use added_T_coef * random_coefs to increase importance of T
            result +=  added_T_coef * random_coefs[-1] *  function_indices[random_fct_idx[-1]](T) # T always selected in the end
            return sigmoid(result + eps)
        

        if created_sites==0:
            sample_size = exp_parameters['host_sample_size']+ exp_parameters['host_test_size']

        else:
            sample_size = np.random.randint(min_sample_size_cand, max_sample_size_cand + 1)  # Add 1 to include max_sample_size_cand

        design_data_cand = subsample_one_dataset(XandT, p_assigned_to_site, sample_size, power_x, power_x_t, outcome_function, std_true_y, seed=np.random.randint(10000))
        design_data_cand = design_data_cand.dropna()
        any_nan = design_data_cand.isna().any().any()
        at_least_20_treated = np.sum(design_data_cand["T"]) > 20
        at_least_20_untreated = len(design_data_cand["T"])-np.sum(design_data_cand["T"]) > 20
        candidate = pd.concat([design_data_cand, data_with_groundtruth.loc[design_data_cand.index, 'Y']], axis=1)

        if binary_outcome:
            at_least_20_y_equal1 = np.sum(candidate["Y"]) > 20
            at_least_20_y_equal0 = len(candidate["Y"])-np.sum(candidate["Y"]) > 20
        else:
            at_least_20_y_equal1 = at_least_20_y_equal0 = True

        if not design_data_cand.empty and not any_nan and at_least_20_treated and at_least_20_untreated and at_least_20_y_equal1 and at_least_20_y_equal0: 
            # we're appending
            candidate_sites[created_sites] = candidate
            created_sites += 1
        else:
            pass # not appending

            
    return candidate_sites

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
    np.random.seed(seed)


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
    np.random.seed(seed)


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

def get_data(dataset: str, path: str) -> tuple:
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
    


    # elif dataset == "acic":
    #     data = pd.read_csv(path + "data/acic_zymu_174570858.csv")
    #     x = pd.read_csv(path + "data/acic_x.csv")
    #     t = data["z"]
    #     y = data["y0"]
    #     idx_to_change = data.loc[data["z"] == 1].index.to_list()
    #     for idx in idx_to_change:
    #         y.loc[idx] = data["y1"].loc[idx]
    #     y = y.rename("y")
    #     one_hot = OneHotEncoder(drop="first").fit(x[["x_2", "x_21", "x_24"]])
    #     new_data = pd.DataFrame(
    #         one_hot.transform(x[["x_2", "x_21", "x_24"]]).toarray(),  # type: ignore
    #     )
    #     x = x.drop(columns=["x_2", "x_21", "x_24"])
    #     x = pd.concat([x, new_data], axis=1)
    else:
        raise ValueError(f"Dataset {dataset} not recognized")
    
    y_std = y.std()
    y_mean = y.mean()
    y = (y - y.mean())/y_std
    data['Y'] = y

    if "y0" in data.columns:
        data[["y0","y1"]] = (data[["y0","y1"]] - y_mean) /y_std

    return data, x, t, y