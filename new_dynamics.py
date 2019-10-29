# ====================================================================================== #
# Module for analyzing dynamical trajectories of conflict avalanches.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from scipy.interpolate import interp1d
import multiprocess as mp
import numpy as np
import pickle


def extract_from_df(subdf, clustersix):
    """Extract the data points necessary from the DataFrame for analysis. These will be
    the time-ordered, reports, fatalities, and locations.

    Parameters
    ----------
    subdf : pandas.DataFrame
    list of lists : indices for the events in conflict avalanches
    cum : bool, True

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame is indexed and sorted in chronological order containing columns for
        fatalities, reports, and cumulative max distances.
    """
    
    from .acled_utils import track_max_pair_dist

    # load data and take all avalanches at each scale
    subdf['SIZES'] = np.ones(len(subdf))
    clusters = []

    for i,cix in enumerate(clustersix):
        # all subsets of subdf corresponding to clusters at this scale
        c = subdf.loc[cix,('EVENT_DATE','FATALITIES','SIZES','LONGITUDE','LATITUDE')]

        # distance is the cum max
        lonlat = c.loc[:,('LONGITUDE','LATITUDE')].values
        if lonlat.ndim==2 and len(lonlat)>1:
            c['DISTANCE'] = track_max_pair_dist(lonlat, False)  # keep track of total distance
        else:
            c['DISTANCE'] = np.zeros(1)
        
        # group by event date (sum simultaneous reports & fatalities but only keep track
        # of max dist)
        c = c[['EVENT_DATE','FATALITIES','SIZES','DISTANCE']]
        gb = c.groupby('EVENT_DATE')  # these are sorted by default
        c = gb.sum()
        c['DISTANCE'] = gb['DISTANCE'].max().values
        c.index = ( (c.index-c.index.min()).values/np.timedelta64(1,'D') ).astype(int)
        c.reset_index(level=0, inplace=True)

        # rename columns
        c = c[['index','FATALITIES','SIZES','DISTANCE']]
        c.columns = 'T','F','S','L'

        clusters.append(c)

    return clusters

def interp_clusters(clusters, x_interp):
    """Loop overconflict avalanche trajectories.

    Parameters
    ----------
    clusters : list
    x_interp : ndarray

    Returns
    -------
    dict
        Regularized data points used to generate interpolated trajectories.
    dict
        Interpolated trajectories.
    """
    
    data = {}
    traj = {}

    data['S'] = regularize_sizes([c[['T','S']].values for c in clusters])
    data['F'] = regularize_fatalities([c[['T','F']].values for c in clusters])
    data['L'] = regularize_diameters([c[['T','L']].values for c in clusters])
    
    traj['S'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['S']])
    traj['F'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['F']])
    traj['L'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['L']])
    return data, traj

def regularize_sizes(listtraj, min_size=3, min_dur=4):
    """Turn sizes trajectory into cumulative profile.

    Parameters
    ----------
    listtraj : list of ndarray
    min_size : int, 3
    min_dur : int, 4

    Returns
    -------
    list of ndarray
    """
    
    reglisttraj = []
    for xy in listtraj:
        if xy[-1,0]>=min_dur and xy[:,1].sum()>=min_size:
            reglisttraj.append(xy.astype(int))
            reglisttraj[-1][:,1] = np.cumsum(reglisttraj[-1][:,1])
            # remove endpoint bias
            reglisttraj[-1][:,1] -= 1
            reglisttraj[-1][-1,1] -= 1

    return reglisttraj

def regularize_fatalities(*args, **kwargs):
    """Turn fatalities trajectory into cumulative profile.

    Parameters
    ----------
    listtraj : list of ndarray
    min_size : int, 2
    min_dur : int, 4

    Returns
    -------
    list of ndarray
    """
    
    return regularize_sizes(*args, **kwargs)

def regularize_diameters(listtraj, min_dur=4, min_diameter=1e-6):
    """Turn diameters trajectory into cumulative profile.

    Parameters
    ----------
    listtraj : list of ndarray

    Returns
    -------
    list of ndarray
    """

    reglisttraj = []
    for xy in listtraj:
        if xy[-1,0]>=min_dur and xy[-1,1]>=min_diameter:
            reglisttraj.append(xy.copy())

    return reglisttraj

def interp(x, y, xinterp):
    """Linear interpolation of trajectories normalized along x and y.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    xinterp : ndarray

    Returns
    -------
    ndarray
    """

    # normalize
    x = x/x[-1]
    y = y/y[-1]
    return interp1d(x, y)(xinterp)
