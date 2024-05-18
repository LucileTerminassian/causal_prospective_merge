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

    if type((kendalltau(implied_ranking, merged_mse)[0])) is float:

        results_dict['tau'] = results_dict.get('tau',[])+[(kendalltau(implied_ranking, merged_mse)[0])] 
        results_dict['rho'] = results_dict.get('rho',[])+[(spearmanr(implied_ranking, merged_mse)[0])]  
    else:
        results_dict['tau'] = results_dict.get('tau',[])+[(kendalltau(implied_ranking, merged_mse)[0]).item()] 
        results_dict['rho'] = results_dict.get('rho',[])+[(spearmanr(implied_ranking, merged_mse)[0]).item()]  
    if type(k) == int:
        results_dict['precision_at_k'] = results_dict.get('precision_at_k',[]) + [precision_at_k(true_cate_ranking, topn_eig_ranking, k=k)]
    else:
        for val in k:
            results_dict['precision_at_'+str(val)] = results_dict.get('precision_at_'+str(val),[]) + [precision_at_k(true_cate_ranking, topn_eig_ranking, k=val)]

    return results_dict

def turn_into_diff(arr):
    n, d = np.shape(arr)[0], np.shape(arr)[1]
    result = np.zeros((n//2, d))
    for i in range (n//2):
        result[i,:]=arr[2*i,:]-arr[(2*i) +1,:]
    return result

from scipy.interpolate import interp1d
from mpl_axes_aligner import align

## below is a plotting function tailored to the illustrative experiment

def plot_dict_illustrative(
    x,
    data_dict: dict,
    axis_names: list,
    mean_color_dict: dict = None,
    std_color_dict: dict = None,
    dict_additional_plots: Union [dict, None] = None,
    text: Union [str, None] = None,
    title: Union[str, None] = None,
    save: Union[str, None] = None,
    second_axis: Union[dict, None] = None,
):

    fig, ax1 = plt.subplots(figsize=(13, 9))

    for label, arr in data_dict.items():

        mean_color = mean_color_dict[label] if mean_color_dict is not None else "blue"
        std_color = std_color_dict[label] if std_color_dict is not None else "blue"

        mean_data = np.mean(arr, axis=0)
        std_data = np.std(arr, axis=0)

        if label.endswith('mcmc'):
            interp_mean_func = interp1d(x, mean_data, kind='linear')
            interp_std_func = interp1d(x, std_data, kind='linear')
            x_interp = np.linspace(min(x), max(x), num=len(mean_data)*3)  # Fine-grained x values for interpolation
            mean_data = interp_mean_func(x_interp)
            std_data = interp_std_func(x_interp)
            ax1.plot(x_interp, mean_data, label=label, color=mean_color, linestyle='--', linewidth=2.2)
            ax1.fill_between(
                x_interp, 
                mean_data - std_data, 
                mean_data + std_data, 
                color=std_color, alpha=0.25,
                linewidth=1.7)
        else:
            ax1.plot(x, mean_data, label=label, color=mean_color, linewidth=2.2)
            ax1.fill_between(x, 
                mean_data - std_data, 
                mean_data + std_data, 
                color=std_color, alpha=0.25, linewidth=2)
            
    if dict_additional_plots is not None:
        for key, arr in dict_additional_plots.items():
            ax1.plot(x, arr, label=key, linewidth=2.0)

    ax1.set_ylabel(axis_names[1], fontsize=25)
    ax1.set_xlabel(axis_names[0], fontsize=33)
    ax1.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.yaxis.set_label_coords(-0.085, 0.5)
    #ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=len(dict.keys())//2, fontsize=20)

    if second_axis:
        ax2 = ax1.twinx() 
        for label, arr in second_axis.items():
            ax2.plot(x, arr, label=label, color='darkblue', linewidth=2.2)
        ax2.set_ylabel(axis_names[2], fontsize=24, rotation=270, labelpad=15)
        ax2.tick_params(axis='y', labelsize=18)
        #ax2.legend(loc='lower left', bbox_to_anchor=(0.14, 0.15), fontsize=20)
        ax2.yaxis.set_label_coords(1.13, 0.5)
    
    ax2.set_yticks([-0.1, -0.05, 0, 0.05])


    # Adjust the plotting range of two y axes
    pos = 0.5  # Position the two origins are aligned
    align.yaxes(ax1, 0, ax2, 0, 0.65)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Define the x-axis ranges for the shaded areas
    shaded_ranges = [(1, 1.25), (1.25, 1.75), (1.75, 2)]

    # Add round text box with label at position (x, y)
    ax1.text(1.15, -0.15, "1", fontsize=20, color='black', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='khaki'))
    ax1.text(1.5, -0.6, "2", fontsize=20, color='black', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='grey'))
    ax1.text(1.87, -0.4, "3", fontsize=20, color='black', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='cornflowerblue'))

    # Define the colors for the shaded areas
    background_colors = ['olive', 'lightgrey', 'lightsteelblue']

    # Add shaded areas to the plot
    for i, (x_start, x_end) in enumerate(shaded_ranges):
        color = background_colors[i]
        if color!= "olive":
            ax1.axvspan(x_start, x_end, color=background_colors[i], alpha=0.3)
        else:
            ax1.axvspan(x_start, x_end, color=background_colors[i], alpha=0.08)
    # Create a single legend with labels from both axes
    fig.legend(handles=handles1 + handles2, labels=labels1 + labels2, \
               loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=24)   
    
    if title is not None:
        fig.suptitle(title)

    if text is not None:
        fig.text(
            0.5, -0.2, text, ha="center", va="center", transform=plt.gca().transAxes
        )

    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    fig.tight_layout()

    if save:
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{save}_{current_date}_{current_time}.pdf"
        fig.savefig(filename, dpi=600, bbox_inches='tight')

    plt.show()