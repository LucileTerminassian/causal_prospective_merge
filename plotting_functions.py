import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime
from typing import Union


def plot_densities(y1, y2, y3, names, title):
    # Create kernel density estimations
    kde_y1 = gaussian_kde(y1)
    kde_y2 = gaussian_kde(y2)
    kde_y3 = gaussian_kde(y3)

    # Plot Y_prior and Y_post_host
    plt.figure(figsize=(10, 6))
    x = np.linspace(-5, 5, 1000)
    plt.plot(x, kde_y1(x), label=names[0], alpha=0.5)
    plt.plot(x, kde_y2(x), label=names[1], alpha=0.5)
    plt.plot(x, kde_y3(x), label=names[2], alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_with_all_sampled_linear(beta_df, X):

    # Don't grab the last column, that is our estimate of the error standard deviation, "sigma"
    if type(beta_df) is pd.core.frame.DataFrame:
        coefficients = beta_df.values
    else:
        coefficients = beta_df
    return coefficients @ X.T


def plot_dict(
    x,
    dict: dict,
    axis_names: list,
    color_dict: dict = None,
    dict_additional_plots: Union [dict, None] = None,
    text: Union [str, None] = None,
    title: Union[str, None] = None,
    save: Union[str, None] = None,
    second_axis: Union[dict, None] = None,
):

    fig, ax1 = plt.subplots(figsize=(13, 8))

    for label, arr in dict.items():
        mean_data = np.mean(arr, axis=0)
        std_data = np.std(arr, axis=0)
        color = color_dict[label] if color_dict is not None else "blue"
        ax1.plot(x, mean_data, label=label, color=color)
        ax1.fill_between(
            x, mean_data - std_data, mean_data + std_data, color=color, alpha=0.3
        )
    if dict_additional_plots is not None:
        for key, arr in dict_additional_plots.items():
            ax1.plot(x, arr, label=key)

    ax1.set_ylabel(axis_names[1], fontsize=12)
    ax1.set_xlabel(axis_names[0], fontsize=20)
    if second_axis is None: 
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(dict.keys()))

    if second_axis:
        ax2 = ax1.twinx() 
        for label, arr in second_axis.items():
            ax2.plot(x, arr, label=label)
        ax2.set_ylabel(axis_names[2], fontsize=12, rotation=270, labelpad=15)

    fig.tight_layout()

    if title is not None:
        fig.suptitle(title)

    if text is not None:
        fig.text(
            0.5, -0.2, text, ha="center", va="center", transform=plt.gca().transAxes
        )
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    if second_axis:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    if save:
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{save}_{current_date}_{current_time}.pdf"
        plt.savefig(filename, dpi=600)

    plt.show()