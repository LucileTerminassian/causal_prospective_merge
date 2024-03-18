import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


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


def plot_array(x, arr, names, title=False):

    n_lines = np.shape(arr)[0]
    plt.figure(figsize=(10, 6))

    for i in range(n_lines):
        plt.plot(x, arr[i, :], label=names[i])

    if title:
        plt.title(title)
    plt.legend()
    plt.show()
