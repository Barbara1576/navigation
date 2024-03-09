import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import MultiPoint, Polygon, Point
from utils.plotting import plot_trajectory
import shapely


def marks2shapely(df_marks, clust_labels, r=3.5, show=True):
    """
    df_marks: df with marks for a mice. clustered
    clust_labels: unique labels of clusters
    r: buffer zone radius

    return:
    marks_poly_list: a triplet of 
    [a marker cluster polygon, buffer zone and label of a cluster]
    """
    marks_poly_list = []

    for lab in clust_labels:
        points = df_marks[['x', 'y']].loc[df_marks['cluster']==lab].to_numpy()
        points = [(xy[0], xy[1]) for xy in points]

        if len(points) == 2:
            point1, point2 = points[0], points[1]
            point3, point4 = (point1[0], point2[1]), (point2[0], point1[1])
            convex_hull_polygon = Polygon([point1, point3,
                                           point2, point4])
        else: 
            multi_point = MultiPoint(points)
            convex_hull_polygon = multi_point.convex_hull
        # Create a buffer
        zone_polygon = convex_hull_polygon.buffer(r)
        marks_poly_list.append([convex_hull_polygon, zone_polygon, lab])
        if show:
            x_zone, y_zone = zone_polygon.exterior.xy
            plt.fill(x_zone, y_zone, alpha=0.25, linewidth=1, linestyle='-',
                    edgecolor='black', facecolor='gray', label='Zone Polygon')

            x, y = convex_hull_polygon.exterior.xy
            plt.fill(x, y, alpha=0.5, linewidth=1, linestyle='-',
                    facecolor='none', edgecolor='black', label='')
            plt.plot(*zip(*points), marker='o', ls='',
                    markersize=4.)
    if show:       
        plt.show()
    
    return marks_poly_list


def plot_intersected_element(intersection):
    if intersection.geom_type == 'MultiLineString':
        for geom in intersection.geoms:
            x, y = geom.xy
            plt.plot(x, y, color='red', linewidth=1.5,
                    alpha=0.2, zorder=4)
    elif intersection.geom_type == 'LineString':
        x, y = intersection.xy
        plt.plot(x, y, color='red', linewidth=1.5,
                alpha=0.2, label='Intersection', zorder=4)


def plot_shapely_traj(trajectory):
    x, y = trajectory.xy
    plt.plot(x, y, color='grey', linewidth=2,
             label='Trajectory', alpha=0.5)
    plt.plot(x, y, color='black',  marker='o', ls='',
             markersize=0.3, alpha=0.5)


def plot_shapely_mark(polygon, poly_buffer, label='none', color='blue'):
    # plot mark poly + buffer
    x_zone, y_zone = poly_buffer.exterior.xy
    plt.fill(x_zone, y_zone, alpha=0.25, linewidth=1, linestyle='-',
                edgecolor='black', facecolor='gray')
    if type(polygon) == LineString:
        coord = shapely.get_coordinates(polygon)
        x, y = coord[:, 0], coord[:, 1]
    elif type(polygon) == Point:
        x, y = polygon.x, polygon.y
    else:
        x, y = polygon.exterior.xy

    plt.fill(x, y, alpha=0.5, linewidth=1, linestyle='-',
            facecolor=color, edgecolor='black', label=label)


def find_traj_marks_intersections(tr, marks_poly_list, show=True):
    """
    tr: df with trajectory and x,y coordinates
    marks_poly_list: 
    """
    points = tr[['x', 'y']].to_numpy()
    trajectory_points = [(xy[0], xy[1]) for xy in points]
    trajectory = LineString(trajectory_points)

    plt.figure(figsize=(7, 7))
    marks_interections = []
    for polygon, poly_buffer, lab in marks_poly_list:

        intersection = polygon.intersection(trajectory)
        intersection_buffer = poly_buffer.intersection(trajectory)
        marks_interections.append([intersection, intersection_buffer, lab])

        # plotting traj, marks and intersections
        if not intersection.is_empty and show:
            plot_intersected_element(intersection)
            plot_shapely_mark(polygon, poly_buffer)
    if show:
        plot_shapely_traj(trajectory)
        plt.show()

    return marks_interections


def extract_shapely_intersections(tr, marks_interections, buff=False,
                                  eps=0.05, show=True):
    tr['intersection'] = 0.
    df_inters_marks = {}
    for intersection, intersection_buff, lab in marks_interections:

        # choose poly_buffer or poly
        intersection = intersection_buff if buff else intersection

        if isinstance(intersection, LineString):
            intersecting_points = list(zip(*intersection.xy))
        elif isinstance(intersection, MultiLineString):
            intersecting_points = [list(zip(*geom.xy)) for geom in intersection.geoms]
            intersecting_points = [point for sublist in intersecting_points for point in sublist]

        indexes = []
        for point in intersecting_points:
            matches = tr[(tr['x'] > point[0]-eps) & (tr['x'] < point[0]+eps) & \
                        (tr['y'] > point[1]-eps) & (tr['y'] < point[1]+eps)]
            indexes.extend(matches.index.tolist())

        # Take uniques idxs
        u_idx = np.sort(np.unique(np.asarray(indexes)))
        print("Unique intersecting points in DataFrame:", len(u_idx)) 
        print(f'Total ratio: {round(len(u_idx) / len(tr), 3)}')

        # Extract from df trajectory parts
        df_inters = tr.loc[u_idx]
        df_inters_marks[lab] = df_inters
            
        # Note intersections in original df
        tr.loc[u_idx, ['intersection']] = lab        

        if show:
            plot_trajectory(df_inters['x'], df_inters['y'], #marker='o',
                            center_mark=False)

    return df_inters_marks, tr