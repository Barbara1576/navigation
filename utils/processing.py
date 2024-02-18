import numpy as np
import pandas as pd
from utils.plotting import (
    plot_trajectory,
    plot_marks_clusters
    )
import pickle
import os
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN


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


def df_minmax(occ):
    if isinstance(occ, pd.DataFrame):
        occ = occ.to_numpy()
    kk = (occ - np.amin(occ)) / (np.amax(occ) - np.amin(occ))
    kl = pd.DataFrame(kk)
    return kl


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


def sort_walls_area(tr, border_tr, wall_percent=0.1): 
    ylen = border_tr['ymax'] - border_tr['ymin'] 
    xlen = border_tr['xmax'] - border_tr['xmin'] 
    xl = border_tr['xmin'] + xlen * wall_percent
    xr = border_tr['xmax'] - xlen * wall_percent
    yd = border_tr['ymin'] + ylen * wall_percent
    yu = border_tr['ymax'] - ylen * wall_percent

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


def add_filtered_V(df, smooth_kern=2):
    dt = df['time'].diff()
    vx = df['x'].diff() / dt
    vy = df['y'].diff() / dt
    v = (vx**2 + vy**2)**0.5

    dt[0], vx[0], vy[0] = 0., 0., 0.
    v[0] = 0.
    if smooth_kern:
        vx = gaussian_filter1d(vx, smooth_kern)
        vy = gaussian_filter1d(vy, smooth_kern)
        v = gaussian_filter1d(v, smooth_kern)
    df['Vx'] = vx
    df['Vy'] = vy
    df['V'] = v
    df.loc[0, 'Vx'] = 0.
    df.loc[0, 'Vy'] = 0.
    df.loc[0, 'V'] = 0.
    return df


def add_step_length(df):
    dx = df['x'].diff()
    dy = df['y'].diff()
    dx[0], dy[0] = 0., 0.,
    step_lens = (dx**2 + dy**2)**0.5
    df['step_length'] = step_lens
    return df


def add_angles(df):
    dt = df['time'].diff()
    vvx = df['x'].diff() / dt 
    vvy = df['y'].diff() / dt 
    dt[0], vvx[0], vvy[0] = 0., 0., 0.

    Xx = np.arctan2(vvx, vvy)
    Nres = normalize_angles_2pi(Xx)
    ang = np.rad2deg(Nres) % 360
    df['angles'] = ang
    return df


def add_delta_angles(df):
    dangles = df['angles'].diff()
    dangles[0] = 0.
    dangles_rad = np.deg2rad(dangles)
    normed_dangels_rad = normalize_angles_2pi(dangles_rad)
    df['delta_angle'] = np.rad2deg(normed_dangels_rad) % 360
    return df


def cut_df(df_to_cut, n_of_parts, t1, t2):
    dur_of_part = (t2 - t1) / n_of_parts
    parts = []
    for i in range(n_of_parts):
        t1_local = t1 + dur_of_part * i
        t2_local = t1 + dur_of_part * (i+1)
        part_df = df_to_cut.loc[(df_to_cut['time'] > t1_local) & (df_to_cut['time'] < t2_local)]
        parts.append(part_df)
    return parts


def cluster_marks(marks_coords, eps=4, min_samples=1,
                  show=False):
    X = np.asarray(marks_coords)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    if show:
        plot_marks_clusters(X, labels)
    return labels


def resample_data_to_larger_timestep(df, original_dt, new_dt):
    from scipy import interpolate

    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    t = df['time'].to_numpy()
    fx = interpolate.interp1d(t, x)
    fy = interpolate.interp1d(t, y)

    new_time = np.arange(df['time'].iloc[0], df['time'].iloc[0] + duration, new_dt)
    newx = fx(new_time)
    newy = fy(new_time)

    resampled_df = pd.DataFrame({'time': new_time,
                                 'x': newx, 
                                 'y': newy})
    return resampled_df