import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime

def plot_densities(y1, y2, y3, names, title):
    # Create kernel density estimations
    kde_y1 = gaussian_kde(y1)
    kde_y2= gaussian_kde(y2)
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

def plot_array(x, arr, axis_names, names, text=False, title=False, save=False):

    n_lines=np.shape(arr)[0]
    plt.figure(figsize=(10, 6))  
    plt.plot(x, arr[0,:], color='blue', label=names[0])
    plt.plot(x, arr[1,:], color='orange', label=names[1])

    if n_lines>2:
        for i in range (2, n_lines):
            if i % 2 == 0:
                plt.plot(x, arr[i,:], color='blue')
            if i % 2 == 1:
                plt.plot(x, arr[i,:], color='orange') 
    
    if title:    
        plt.title(title)
    plt.ylabel(axis_names[1])
    plt.xlabel(axis_names[0])
    plt.legend()

    if text:
        plt.text(0.5, -0.2, text, ha='center', va='center', transform=plt.gca().transAxes)

    if save:
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{save}_{current_date}_{current_time}.pdf"
        plt.savefig(filename)

    plt.show()