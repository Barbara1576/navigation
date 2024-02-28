import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy import stats
from utils.processing import df_minmax, rescale_coord, minmax_normalize

from scipy.signal import butter, filtfilt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_trajectory(x, y, xylim=None, center_mark=True):
    time_cmap = np.arange(0, len(x))
    plt.scatter(x, y, s=1, c=time_cmap)
    plt.plot(x, y, c='gray', alpha=0.4)
    if center_mark:
        plt.scatter(0, 0, marker='X', s=60, c='red')
    plt.colorbar()
    if xylim:
        plt.xlim(xylim)
        plt.ylim(xylim)
    plt.show()


def plot_wall_sorted_samples(df, plot_title):
    # hard-code areas labels :)
    all_areas = [40, 30, 20, 10, 4, 3, 2, 1, 0]
    for a in all_areas:
        x = df.loc[df['near_wall'] == a]['x']
        y = df.loc[df['near_wall'] == a]['y']
        plt.scatter(x, y, s=1.5, label=str(a))
    plt.gca().set_aspect('equal')
    plt.title(plot_title)
    plt.legend(loc='center')
    plt.show()


def plot_marks_clusters(X, labels):

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ = list(labels).count(-1)

    plt.figure(figsize=(6, 6))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # black used for noise
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def occupMapNorm(df, binn, eps=0.1):
    """ 
    делаем df где будет понятно сколько было точек в каждом бине 
    соответсвуеющему данному интервалу х и данному интервалу у
    """
    xmin = df["x"].min() - eps
    xmax = df["x"].max() + eps
    binx = np.linspace(xmin, xmax, binn)
    x_cut = pd.cut(df.x, binx, right=False)

    ymin = df["y"].min() - eps
    ymax = df["y"].max() + eps
    biny = np.linspace(ymin, ymax, binn)
    y_cut = pd.cut(df.y, biny, right=False)

    dfn = df.groupby([x_cut, y_cut], observed=False).count()
    
    dfn = dfn.drop(columns=['x', 'y'])
    dfn = dfn.rename(columns={'time': 'Count'})
    
    L = [(a.mid, b.mid) for a, b in dfn.index]
    dfm = dfn.set_index(pd.MultiIndex.from_tuples(L, names=dfn.index.names))
    
    dd = dfm.unstack()
    arrd = dd.replace(0., 1).to_numpy()
    arrl = np.log(arrd)

    arrl = df_minmax(arrl)
    return arrl


def plot_occupancy_plot(df, n_bins=30, sigma=1,
                        ax=None, fig=None):
    df_copy = df[['time', 'x', 'y']]
    occ = occupMapNorm(df_copy, binn=n_bins)

    occ_s = gaussian_filter(occ, sigma=sigma)
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(occ_s, cmap='jet', interpolation='none') 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    if not ax:
        plt.show()


def smooth_data(sig, N=6, Wn=0.15):
    # param to change: the lower Wn, the smoother traj 
    b, a = butter(N, Wn, 'low')
    sig_smoothed = filtfilt(b, a, sig)
    return sig_smoothed


def calc_velocity_and_speed(xx, yy):
    vx = np.diff(xx, append=xx[-1])
    vy = np.diff(yy, append=yy[-1])
    s = np.sqrt(vx**2 + vy**2)
    return vx, vy, s


def calc_acceleration(vx, vy):
    ax = np.diff(vx, append=vx[-1])
    ay = np.diff(vy, append=vy[-1])
    a = np.sqrt(ax**2 + ay**2)
    return ax, ay, a


def calc_curvature(vx, vy, ax, ay):
    c = (vx * ay - vy * ax) / ((np.power(vx**2 + vy**2, 1.5)) + 1e-3)
    return c


def plot_trajectory_with_median(xx, yy):
    """
    Calculate curvature to speed ratio and find indices
    for lower and higher than median values
    """
    vx, vy, s = calc_velocity_and_speed(xx, yy)
    ax, ay, a = calc_acceleration(vx, vy)
    c = calc_curvature(vx, vy, ax, ay)

    c_nonzero = np.where(c==0, 1e-3, c)
    s_nonzero = np.where(s==0, 1e-3, s)

    c_to_s_ratio = np.abs(c_nonzero / s_nonzero)    # NOTE: another way to define stops?
    median_c_to_s_ratio = np.median(c_to_s_ratio)
    II_low = np.where(c_to_s_ratio < median_c_to_s_ratio)[0]
    II_high = np.where(c_to_s_ratio >= median_c_to_s_ratio)[0]

    # Plot segments with lower than median ratio in red
    plt.plot(xx[II_low], yy[II_low], '.r', markersize=0.7)
    # Plot segments with higher than median ratio in blue
    plt.plot(xx[II_high], yy[II_high], '.b', markersize=0.7)
    plt.plot(xx, yy, 'grey', alpha=0.2, lw=0.7)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_vector_field(xx, yy, L=15, n_bins=16,
                      ax=None, fig=None):
    """
    Implementation of Lebedev et al.
    L: Radius for neighborhood calculations
    """
    NN, xedges, yedges = np.histogram2d(xx, yy, bins=n_bins)
    X, Y, U, V, Densities = [], [], [], [], []
    vx, vy, s = calc_velocity_and_speed(xx, yy)
    for ix in range(len(xedges)-1):
        for iy in range(len(yedges)-1):
            x0 = (xedges[ix] + xedges[ix+1]) / 2
            y0 = (yedges[iy] + yedges[iy+1]) / 2
            
            mask = ((xx - x0) ** 2 + (yy - y0) ** 2) < L**2
            selected_vx = vx[mask]
            selected_vy = vy[mask]
            
            if len(selected_vx) > 0:
                d = np.mean(selected_vx + 1j * selected_vy)
                d /= np.abs(d) ** 0.6
                X.append(x0)
                Y.append(y0)
                U.append(np.real(d))
                V.append(np.imag(d))
                Densities.append(len(selected_vx))
    max_density = max(Densities)
    min_density = min(Densities)
    normalized_densities = [(d - min_density) / (max_density - min_density) for d in Densities]
    colors = [cm.Reds(density) for density in normalized_densities]

    if not ax:
        fig, ax = plt.subplots(figsize=(8, 6))
    magnitude = np.sqrt(np.asarray(U)**2 + np.asarray(V)**2)
    magnitude[magnitude == 0] = 1
    U_normalized = U / magnitude
    V_normalized = V / magnitude
    q = ax.quiver(X, Y, U_normalized, V_normalized,
                color=colors, 
                scale_units='xy', 
                scale=0.3,
                angles='xy')
    sm = cm.ScalarMappable(cmap=cm.Reds,
                           norm=Normalize(vmin=min_density,
                                          vmax=max_density))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax,
                        label='Density of Data Points')
    if not ax:
        plt.show()


def speed_z_score_norm(vx, vy, mean, sd, eps=1e-2):
    vb = ((vx ** 2) + (vy ** 2)) ** 0.5
    vn = abs((vb - mean) / (sd + eps))

    vx = vn * vx / (vb + eps)
    vy = vn * vy / (vb + eps)
    return vx, vy


def kill_nans(x, grid_size, verbose=False):
    for i in range(grid_size):
        for j in range(grid_size):
            if np.isnan(x.iloc[i, j]):
                if verbose:
                    print(f'NaN in place: {i}, {j}')
                idx_list = np.array([[i-1, j-1], [i-1, j], [i-1, j+1],
                                        [i, j-1]  ,           [i, j+1],
                                        [i+1, j-1], [i+1, j], [i+1, j+1]])
                
                idx_to_drop = np.where((idx_list < 0) | (idx_list > grid_size-1))[0]
                if verbose:
                    print(f'Dropping indexes: {idx_to_drop}')
                dropping_mask = np.ones(idx_list.shape[0], dtype=bool)
                dropping_mask[idx_to_drop] = False
                idx_list = idx_list[dropping_mask]

                x.iloc[i, j] = np.nanmean(x.iloc[idx_list[:, 0], idx_list[:, 1]])
    return x


def build_vector_field(df, grid_size, title_name, z_norm,
                       lscale=0.17, sigma=2, color_map='Reds',
                       eps=1e-2, unsmoothed_colors=False, save=False):
    """
    Our old implementation of vector field
    """
    df = df.reset_index(drop=True)
    binx = np.linspace(df["x"].min() - eps, df["x"].max() + eps,
                       grid_size+1)
    x_binned = pd.cut(df.x, binx, right=False)
    biny = np.linspace(df["y"].min() - eps, df["y"].max() + eps,
                       grid_size+1)
    y_binned = pd.cut(df.y, biny, right=False)

    df_binned_mean = df.groupby([x_binned, y_binned],
                                 observed=False).mean()
    dfvx = pd.DataFrame(df_binned_mean.loc[:, 'Vx']).unstack()
    dfvy = pd.DataFrame(df_binned_mean.loc[:, 'Vy']).unstack()

    # For z_normalization
    if z_norm:
        df_binned_sd = df.groupby([x_binned, y_binned],
                                observed=False).std()
        mV = pd.DataFrame(df_binned_mean.loc[:, 'V']).unstack()
        sdV = pd.DataFrame(df_binned_sd.loc[:, 'V']).unstack()
        mV =  kill_nans(mV, grid_size, verbose=False)
        sdV = kill_nans(sdV, grid_size, verbose=False)

    # Average Nans with all neightboring values
    dfvx = kill_nans(dfvx, grid_size)
    dfvy = kill_nans(dfvy, grid_size)

    vector_field_matrix = []
    for i in range(grid_size):
        vector_column = []
        for j in range(grid_size):
            if z_norm:
                vx, vy = speed_z_score_norm(dfvx.iloc[i, j], dfvy.iloc[i, j],
                                            mV.iloc[i, j], sdV.iloc[i, j]) 
            else: 
                vx = dfvx.iloc[i, j]
                vy = dfvy.iloc[i, j]

            vector_column.append([binx[i], biny[j], vx, vy])
        vector_field_matrix.append(vector_column)

    vector_field_matrix = np.array(vector_field_matrix)

    vectx = vector_field_matrix[:, :, 2]
    vecty = vector_field_matrix[:, :, 3]

    vectx_smoothed = gaussian_filter(vectx, sigma=sigma)
    vecty_smoothed = gaussian_filter(vecty, sigma=sigma)

    L_matrix = (vectx_smoothed**2 + vecty_smoothed**2)**0.5
    L_matrix_unsmoothed = (vectx**2 + vecty**2)**0.5 

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.axvline(0, c='black', alpha=0.2)
    ax.axhline(0, c='black', alpha=0.2)
    
    L = L_matrix_unsmoothed if unsmoothed_colors else L_matrix
    min_val, max_val = np.min(L), np.max(L)
    cmap = plt.cm.get_cmap(color_map)
    color_list = cmap(L)

    for i in range(grid_size):
        for j in range(grid_size):
            l = L_matrix[i, j]
            im1 = ax.quiver(vector_field_matrix[i, j, 0],
                            vector_field_matrix[i, j, 1],
                            vectx_smoothed[i, j], vecty_smoothed[i, j],
                            color=color_list[i, j], units='xy', pivot='middle',
                            scale=lscale*l, width=0.6)
    plt.title(f"Vector Field of Speed, {title_name}")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=min_val, vmax=max_val)
    plt.colorbar(sm, ax=ax)
    if save:
        plt.savefig(os.path.join(os.getcwd(), 'images',
                                 title_name + '_NewVelocityField.png'), dpi=150)
    plt.show()


def plot_divergence(xx, yy, n_bins=30, level=35, scaling=True,
                    ax=None, fig=None):
    Vx = np.gradient(xx)
    Vy = np.gradient(yy)

    N, x_edges, y_edges = np.histogram2d(xx, yy,
                                         bins=n_bins,
                                         density=True)
    # Create a 2D grid for the velocity components
    X, Y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)

    Vx_grid = griddata((xx, yy), Vx, (X, Y), method='cubic', fill_value=0)
    Vy_grid = griddata((xx, yy), Vy, (X, Y), method='cubic', fill_value=0)

    Vx_grad_x, Vx_grad_y = np.gradient(Vx_grid, axis=(1, 0))
    Vy_grad_x, Vy_grad_y = np.gradient(Vy_grid, axis=(1, 0))

    divergence = Vx_grad_x + Vy_grad_y
    # Scale with sqrt and keep sign
    if scaling:
        divergence = rescale_coord(divergence, new_min=-1, new_max=1)
        divergence = np.sign(divergence) * np.sqrt(np.abs(divergence))

    if not ax:
        fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.contourf(X, Y, divergence, 
                    levels=level,  # increase this param to "interpolate"
                    cmap='RdYlBu_r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Divergence Map')
    if not ax:
        plt.show()

