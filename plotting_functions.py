import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

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
    coefficients = beta_df.values
    
    array_list = []
    # Find our linear combination again
    for i in range (len(coefficients)):
        array_list.append(pd.DataFrame(X.dot(coefficients[i,:]), index= X.index))
    
    return np.hstack(array_list).T