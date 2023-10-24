import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import math
import scipy


def open_file(fpath, show=False):
    df = pd.read_csv(fpath, sep=" ", header = None, names=['numb', 'time', 'x', 'y'])
    df = df.drop(df[df.time > 1755].index)
    if show:
        x = df['x'].values.tolist()
        y = df['y'].values.tolist()
        plot_trajectory(x, y)
    return df


def plot_trajectory(x, y):
    time_cmap = np.arange(0, len(x))
    plt.scatter(x, y, s=1, c=time_cmap)
    plt.plot(x, y, c='gray', alpha=0.4)
    plt.scatter(0, 0, marker='X', s=60, c='red')
    plt.colorbar()
    plt.show()