import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import math
import scipy


def plot_trajectory(x, y, xylim=None):
    time_cmap = np.arange(0, len(x))
    plt.scatter(x, y, s=1, c=time_cmap)
    plt.plot(x, y, c='gray', alpha=0.4)
    plt.scatter(0, 0, marker='X', s=60, c='red')
    plt.colorbar()
    if xylim:
        plt.xlim(xylim)
        plt.ylim(xylim)
    plt.show()


# TODO: add def for plot dots in areas 