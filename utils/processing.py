import numpy as np
import pandas as pd
from utils.plotting import plot_trajectory
import pickle
import os


def open_file(ff, show=True):
    df = pd.read_csv(ff, sep=" ", header=None, names=['idx', 'time', 'x', 'y'])
    df = df.drop('idx', axis=1)
    df = df.drop(df[df.time > 1755].index)
    if show:
        plot_trajectory(df['x'], df['y'])
    return df


def normalize_angles_2pi(y):
    exceeded_indices = np.where((y < 0.))[0]
    y[exceeded_indices] += 2*np.pi

    exceeded_indices = np.where(y > 2*np.pi)[0]
    if len(exceeded_indices) > 0:
        y[exceeded_indices] = np.fmod(y[exceeded_indices], 2*np.pi)
    return y


def cut_jumps(df, xmax, xmin, ymax, ymin, show=True):
    """
    Simply drops samples with jumps onto the walls
    """
    dff = df[(df['x'] < xmax) & (df['x'] > xmin) & (df['y'] < ymax) & (df['y'] > ymin)]
    ind = range(0, len(dff))
    dff = dff.reindex(ind)
    if show:
        plot_trajectory(dff['x'], dff['y'])
    return dff


def rescale_coord(data, new_min, new_max):
    min_old_x, max_old_x = np.min(data), np.max(data)
    data_rescaled = ((data - min_old_x) / (max_old_x - min_old_x)) * (new_max - new_min) + new_min
    return data_rescaled


def save_preprocessed_data(sub_trajs_list, k, PATH):
    if type(sub_trajs_list) != list:
        raise ValueError('Should be a list!')

    fname = os.path.join(PATH, f'{k}.pkl')
    with open(fname, 'wb') as outp:
        pickle.dump(sub_trajs_list, outp, pickle.HIGHEST_PROTOCOL)
    print(f'Successfully saved {k} as {fname}')


def open_preprocessed_data(fname):
    with open(fname, 'rb') as inp:
        sub_trajs_list = pickle.load(inp)
    return sub_trajs_list


def sort_walls_area(tr, wall_percent=0.1): 
    ylen = tr['y'].max() - tr['y'].min() 
    xlen = tr['x'].max() - tr['x'].min() 
    xl = tr['x'].min() + xlen * wall_percent
    xr = tr['x'].max() - xlen * wall_percent
    yd = tr['y'].min() + ylen * wall_percent
    yu = tr['y'].max()  - ylen * wall_percent

    near_wall_list = np.zeros((len(tr),))
    for i in range(len(tr)):     
        if tr['x'].iloc[i] < xl:
            if tr['y'].iloc[i] > yu:
                near_wall_list[i] = 40
            elif tr['y'].iloc[i] < yd:
                near_wall_list[i] = 30
            else:
                near_wall_list[i] = 4
        elif tr['x'].iloc[i] > xr:
            if tr['y'].iloc[i] > yu:
                near_wall_list[i] = 10
            elif tr['y'].iloc[i] < yd:
                near_wall_list[i] = 20
            else:
                near_wall_list[i] = 2
        elif tr['y'].iloc[i] > yu:
            near_wall_list[i] = 1
        elif tr['y'].iloc[i] < yd:
            near_wall_list[i] = 3

    tr['near_wall'] = np.asarray(near_wall_list)
    return tr