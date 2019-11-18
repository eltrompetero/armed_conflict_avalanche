# ====================================================================================== #
# Module for analyzing dynamical trajectories of conflict avalanches.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from scipy.interpolate import interp1d
import multiprocess as mp
import numpy as np
import pickle
import pandas as pd


def extract_from_df(subdf, clustersix, run_checks=False, null_type=None):
    """Extract the data points necessary from the DataFrame for analysis. These will be
    the time-ordered, day-by-day reports, fatalities, and locations.

    Reports and fatalities are summed by day. Max locations are saved.

    Parameters
    ----------
    subdf : pandas.DataFrame
    clustersix : list of lists
        Indices for the events in conflict avalanches
    run_checks : bool, False
    null_type : str, None
        'permute': Shuffle the order of the events while maintaining the dates on which
            events happened..
        'uniform': Smear events randomly and uniformly over entire interval of conflict
            cluster.
        'flat': Smear events uniformly through time and enforce equal numbers of events
            per day (this second part is not done for diameters, a point that would
            require some more complicated gynamstics).
        'completelyflat': Set a single event per day between first and last days. More for
            a sanity check.

    Returns
    -------
    list of pd.DataFrame
        Each DataFrame is indexed and sorted in chronological order containing columns for
        fatalities, reports, and cumulative max distances.
    """
    
    from .acled_utils import track_max_pair_dist

    totalf = 0
    totals = 0

    # load data and take all avalanches at each scale
    # sizes meant to keep track of the number of reports
    subdf['SIZES'] = np.ones(len(subdf))
    clusters = []

    for i,cix in enumerate(clustersix):
        # all subsets of subdf corresponding to clusters at this scale
        c = subdf.loc[cix,('EVENT_DATE','FATALITIES','SIZES','LONGITUDE','LATITUDE')]
        if null_type=='permute':
            # permute measurements of size together
            randix = np.random.permutation(range(len(c)))
            c.iloc[:,1:] = c.iloc[randix,1:].values
        elif null_type=='uniform' and len(cix)>2:  # can't do this with very short samples
            # uniformly sample days in between first and last events, ensuring that first and last dates are
            # maintained and keeping the same total number of days
            dt = int((c['EVENT_DATE'].max()-c['EVENT_DATE'].min())/np.timedelta64(1,'D'))
            randdates = c['EVENT_DATE'].min() + np.array([pd.Timedelta(days=i)
                                                          for i in np.random.randint(dt+1, size=len(c)-2)])
            c.iloc[0,0] = c['EVENT_DATE'].min()
            c.iloc[-1,0] = c['EVENT_DATE'].max()
            c.iloc[1:-1,0] = np.sort(randdates)

            # permute measurements of size together
            randix = np.random.permutation(range(len(c)))
            c.iloc[:,1:] = c.iloc[randix,1:].values
        elif null_type=='flat' and len(cix)>2:
            # equally space all events out over all the observed days 
            dt = int((c['EVENT_DATE'].max()-c['EVENT_DATE'].min())/np.timedelta64(1,'D'))
            randdates = c['EVENT_DATE'].min() + np.array([pd.Timedelta(days=i)
                                                          for i in np.random.randint(dt+1, size=len(c)-2)])
            c.iloc[0,0] = c['EVENT_DATE'].min()
            c.iloc[-1,0] = c['EVENT_DATE'].max()
            c.iloc[1:-1,0] = np.sort(randdates)

            # apportion all events equally over all days
            c.iloc[:,1:3] = c.iloc[:,1:3].values.sum(0)/len(c)
        elif null_type=='completelyflat':
            # equally space all events out over the entire range of days between first and last
            dt = int((c['EVENT_DATE'].max()-c['EVENT_DATE'].min())/np.timedelta64(1,'D'))
            dates = c['EVENT_DATE'].min() + np.array([pd.Timedelta(days=i)
                                                      for i in range(dt+1)])
            data = np.hstack((dates[:,None], np.ones((dt+1,4))))
            c = pd.DataFrame({'EVENT_DATE':dates,
                              'FATALITIES':np.ones(dt+1),
                              'SIZES':np.ones(dt+1),
                              'LONGITUDE':np.random.choice(c['LONGITUDE'].values, size=dt+1),
                              'LATITUDE':np.random.choice(c['LATITUDE'].values, size=dt+1)})

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
        if run_checks:
            totalf += c['F'].sum()
            totals += c['S'].sum()

    if run_checks:
        # check that days on which events happened increase monotonically
        assert all([(np.sign(np.diff(c['T']))>0).all() for c in clusters])
        # check that total number of fatalities matches up
        assert subdf['FATALITIES'].sum()==totalf
        # check that total number of reports matches up
        assert subdf['SIZES'].shape[0]==totals

    return clusters

def interp_clusters(clusters, x_interp, method=None):
    """Loop overconflict avalanche trajectories.

    Parameters
    ----------
    clusters : list
    x_interp : ndarray
    method : str, None
        Force particular method on all trajectories.
        'piecewise'
        'linear'

    Returns
    -------
    dict
        Regularized data points used to generate interpolated trajectories.
    dict
        Interpolated trajectories.
    """
    
    data = {}
    traj = {}
    clusterix = {}  # indices of clusters kept for analysis and not removed because they
                    # didn't meet the cutoff

    data['S'], clusterix['S'] = regularize_sizes([c[['T','S']].values for c in clusters])
    # bias in average fatality trajectories has to be accounted for later (on the ensemble average)
    data['F'], clusterix['F'] = regularize_fatalities([c[['T','F']].values for c in clusters])
                                                      #[1/c['S'].sum() for c in clusters])
    # according to old dynamics code, you should subtract the average fraction across all avalanches
                                      #[np.mean([1/c['S'].sum() for c in clusters if c['S'].sum()>2])]*len(clusters))
    data['L'], clusterix['L'] = regularize_diameters([c[['T','L']].values for c in clusters])
    
    if method=='piecewise':
        traj['S'] = np.vstack([piecewise_interp(xy[:,0], xy[:,1], x_interp) for xy in data['S']])
        traj['F'] = np.vstack([piecewise_interp(xy[:,0], xy[:,1], x_interp) for xy in data['F']])
        traj['L'] = np.vstack([piecewise_interp(xy[:,0], xy[:,1], x_interp) for xy in data['L']])
    elif method=='linear':
        traj['S'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['S']])
        traj['F'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['F']])
        traj['L'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['L']])
    else:
        traj['S'] = np.vstack([piecewise_interp(xy[:,0], xy[:,1], x_interp) for xy in data['S']])
        traj['F'] = np.vstack([piecewise_interp(xy[:,0], xy[:,1], x_interp) for xy in data['F']])
        traj['L'] = np.vstack([interp(xy[:,0], xy[:,1], x_interp) for xy in data['L']])

    return data, traj, clusterix


# ================ #
# Helper functions #
# ================ #
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
    list of ints
        Indices of trajectories that were kept. These correspond to the indices of the
        clusters.
    """
    
    reglisttraj = []
    keepix = []

    for i,xy in enumerate(listtraj):
        if xy[-1,0]>=min_dur and xy[:,1].sum()>=min_size:
            reglisttraj.append(xy.astype(int))
            reglisttraj[-1][:,1] = np.cumsum(reglisttraj[-1][:,1])
            # remove endpoint bias
            reglisttraj[-1][:,1] -= 1
            reglisttraj[-1][-1,1] -= 1
            keepix.append(i)

    return reglisttraj, keepix

def regularize_fatalities(listtraj, fraction_bias=None, min_size=3, min_dur=4):
    """Turn fatalities trajectory into cumulative profile.

    Way of regularizing involves subtracting a mean bias across all trajectories which
    means that individual trajectories might be negative but the mean should be
    reasonable.

    Parameters
    ----------
    listtraj : list of ndarray
    fraction_bias : list, None
    min_size : int, 3
    min_dur : int, 4

    Returns
    -------
    list of ndarray
    list of ints
        Indices of trajectories that were kept. These correspond to the indices of the
        clusters.
    """

    reglisttraj = []
    keepix = []
    if fraction_bias is None:
        fraction_bias = [0]*len(listtraj)

    for i,(xy,fb) in enumerate(zip(listtraj,fraction_bias)):
        if xy[-1,0]>=min_dur and xy[:,1].sum()>=min_size:
            reglisttraj.append(xy.astype(float))
            reglisttraj[-1][:,1] = np.cumsum(reglisttraj[-1][:,1])
            # remove endpoint bias (avg no. of fatalities per report)
            reglisttraj[-1][:,1] -= fb * reglisttraj[-1][-1,1]
            reglisttraj[-1][-1,1] -= fb * reglisttraj[-1][-1,1]
            keepix.append(i)
    
    return reglisttraj, keepix

def regularize_diameters(listtraj, min_dur=4, min_diameter=1e-6):
    """Turn diameters trajectory into cumulative profile.

    Parameters
    ----------
    listtraj : list of ndarray

    Returns
    -------
    list of ndarray
    list of ints
        Indices of trajectories that were kept. These correspond to the indices of the
        clusters.
    """

    reglisttraj = []
    keepix = []

    for i,xy in enumerate(listtraj):
        if xy[-1,0]>=min_dur and xy[-1,1]>=min_diameter:
            reglisttraj.append(xy.copy())
            keepix.append(i)

    return reglisttraj, keepix

def size_endpoint_bias(clusters, clusterix):
    """Endpoint bias for fatality trajectories assuming that fatalities are distributed
    uniformly across all reports such that the bias is just given by the average reports
    bias.

    Parameters
    ----------
    clusters : list of pd.DataFrame
    clusterix : list of ints
        This is returned by extract_from_df().

    Returns
    -------
    float
        Bias at endpoints induced by definition of conflict avalanches by where there is
        at least one conflict report.
    """

    return (1/np.array([clusters[i]['S'].sum() for i in clusterix])).mean()

def keep_fat_of_dur(dataF, mn, mx):
    """Return indices of elements for fatalities that satisfy mn <= f < mx.

    Remember that fatalities durations are off by one because counting starts at t=0.

    Parameters
    ----------
    dataF : list of ndarrays
        Fatalities trajectories from dict "data".
    mn : int
    mx : int
    
    Returns
    -------
    list of int
        Indices of fatalities that satisfy duration bounds.
    """

    assert 0<mn<mx
    assert mx<=8192

    return [i for i,traj in enumerate(dataF) if mn<=traj[-1,0]<mx]

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
    return interp1d(x, y, bounds_error=False)(xinterp)

def piecewise_interp(x, y, xinterp, **kwargs):
    """Linear interpolation of trajectories normalized along x and y and with piecewise
    interpretation of relationship between x and y.

    Curves as defined such that the value after the jump occurs at the end of the flat
    segment (this is like the right-handed CDF).

    Parameters
    ----------
    x : ndarray
        Vector of integers. Must be sorted.
    y : ndarray
    xinterp : ndarray

    Returns
    -------
    ndarray
        Interpolated trajectory.
    """
    
    # normalize
    x = np.insert(x, range(1,x.size),  x[1:])
    x = x/x[-1]
    y = np.insert(y, range(y.size-1), y[:-1])
    y = y/y[-1]

    # flat step at the end
    #x = np.append(x, 1)
    #y = np.append(y, 1)
    return interp1d(x, y, assume_sorted=True, **kwargs)(xinterp)
