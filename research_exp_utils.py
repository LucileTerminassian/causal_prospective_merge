import numpy as np
import pandas as pd

from rct_data_generator import *
from outcome_models import *
from plotting_functions import *
from mcmc_bayes_update import *
from eig_comp_utils import *
from scipy.stats import kendalltau, spearmanr

############################ CLOSED FORM


def linear_eig_closed_form_varying_sample_size(
    data: dict[int, dict],
    data_parameters: dict[str],
    prior_hyperparameters: dict[str],  # passed to BayesianLinearRegression
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
    data_parameters: dict[str],
    prior_hyperparameters: dict[str],
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
        print('length is '+str(length))
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
            print('cand is '+str(cand))
            X_cand = torch.from_numpy(dlen[cand].drop(columns=["Y"]).values)
            if verbose:
                print(f" % treated in {cand}: {int(100 * dlen[cand]['T'].mean())}%")

            EIG_obs[cand].append(bayes_reg.samples_obs_EIG(
                X_cand, n_samples_outer_expectation_obs, n_samples_inner_expectation_obs))
            
            print("obs done")
            EIG_caus[cand].append(bayes_reg.samples_causal_EIG(
                X_cand, n_samples_outer_expectation_caus, n_samples_inner_expectation_caus))
            print("caus done")

    return EIG_obs, EIG_caus


def bart_eig_from_samples_varying_sample_size(
    
    data: dict[int, dict],
    data_parameters: dict[str],
    prior_hyperparameters: dict[str],
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



def average_precision_at_k(true_rankings, predicted_rankings, k):
    num_hits = 0
    sum_precision = 0
    for i, pred in enumerate(predicted_rankings[:k], 1):
        if pred in true_rankings:
            num_hits += 1
            sum_precision += num_hits / i
    if not true_rankings:
        return 0
    return sum_precision / min(len(true_rankings), k)

def mean_average_precision(true_rankings, predicted_rankings, k=None):
    if k is None:
        k = len(true_rankings)
    avg_precision = np.mean([average_precision_at_k(true_rankings, predicted_rankings, k_) for k_ in range(1, k + 1)])
    return avg_precision

def precision_at_k(true_rankings, predicted_rankings, k):
    intersection = set(predicted_rankings[:k]) & set(true_rankings[:k])
    return len(intersection) / k

def recall_at_k(true_rankings, predicted_rankings, k):
    intersection = set(predicted_rankings[:k]) & set(true_rankings[:k])
    return len(intersection) / len(true_rankings)

def mrr(true_rankings, predicted_rankings):
    for i, pred in enumerate(predicted_rankings, 1):
        if pred in true_rankings:
            return 1 / i
    return 0

def ndcg(true_rankings, predicted_rankings, k=None):
    if k is None:
        k = len(true_rankings)
    dcg = sum(2 ** true_rankings[i] - 1 / np.log2(i + 2) for i in range(k))
    ideal_rankings = sorted(true_rankings, reverse=True)
    ideal_dcg = sum(2 ** ideal_rankings[i] - 1 / np.log2(i + 2) for i in range(k))
    return dcg / ideal_dcg


def compare_to_ground_truth(results_dict, true_cate_ranking, eig_ranking, merged_mse, top_n = None, k = None):
    
    if top_n is not None:
        topn_eig_ranking = eig_ranking[:top_n]
        topn_true_cate_ranking = true_cate_ranking[:top_n]
        topn_merged_mse = merged_mse[:top_n]
    else: 
        topn_eig_ranking, topn_true_cate_ranking,topn_merged_mse = eig_ranking, true_cate_ranking,merged_mse

    if k is None:
        k = len(true_cate_ranking)
    
    implied_ranking = [eig_ranking.index(val) for val in list(range(min(eig_ranking),max(eig_ranking)+1))]
    
    print(kendalltau(implied_ranking, merged_mse)[0])
    results_dict['tau'] = results_dict.get('tau',[])+[(kendalltau(implied_ranking, merged_mse)[0]).item()] #.item()
    results_dict['rho'] = results_dict.get('rho',[])+[(spearmanr(implied_ranking, merged_mse)[0]).item()]  #.item()

    if type(k) == int:
        results_dict['precision_at_k'] = results_dict.get('precision_at_k',[]) + [precision_at_k(true_cate_ranking, topn_eig_ranking, k=k)]
    else:
        for val in k:
            results_dict['precision_at_'+str(val)] = results_dict.get('precision_at_'+str(val),[]) + [precision_at_k(true_cate_ranking, topn_eig_ranking, k=val)]
    
    # results_dict['recall_at_k'].append(recall_at_k(true_cate_ranking, topn_eig_ranking, k=k[0]))
    # if type(k) == int:
    #     results_dict['recall_at_k'].append(recall_at_k(true_cate_ranking, topn_eig_ranking, k=k))
    # else:
    #     for val in k:
    #         results_dict['recall_at_'+str(val)].append(recall_at_k(true_cate_ranking, topn_eig_ranking, k=val))
    # results_dict['mean average precision'].append(mean_average_precision(topn_true_cate_ranking, topn_eig_ranking, k=k[0]))
    # results_dict['ndcg'].append(ndcg(topn_true_cate_ranking, topn_eig_ranking, k[0]))
    # results_dict['rank corr eig'].append(np.corrcoef(topn_true_cate_ranking, topn_eig_ranking)[0, 1])
    # results_dict['mean reciprocal rank'].append(mrr(topn_true_cate_ranking, topn_eig_ranking))

    return results_dict