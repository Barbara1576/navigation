import numpy as np


def normalize(y):
    exceeded_indices = np.where((y < 0.))[0]
    y[exceeded_indices] += 2*np.pi

    exceeded_indices = np.where(y > 2*np.pi)[0]
    if len(exceeded_indices) > 0:
        y[exceeded_indices] = np.fmod(y[exceeded_indices], 2*np.pi)
    return y