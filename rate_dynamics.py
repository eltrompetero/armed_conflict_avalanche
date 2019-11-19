# ====================================================================================== #
# Module for analyzing dynamical trajectories of conflict avalanches (see
# cum_dyanmics.py). 
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .cum_dynamics import *


def interp_clusters(clusters, x_interp, bins, npoints, method=None):
    """Loop overconflict avalanche trajectories.

    Parameters
    ----------
    clusters : list
    x_interp : ndarray
    bins : list
    npoints : list
        Of length len(bins)-1.
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

    data['S'], clusterix['S'] = regularize_sizes([c[['T','S']].values for c in clusters], bins, npoints)
    data['F'], clusterix['F'] = regularize_fatalities([c[['T','F']].values for c in clusters], bins, npoints)
    data['L'], clusterix['L'] = regularize_diameters([c[['T','L']].values for c in clusters], bins, npoints)
    
    # interpolate avg
    # create tuples where first element is just the averages on the discretized time points and the second is
    # the interpolated rate profiles (like what would be plotted if a line were drawn between the points
    # without having accounted for the density to generate accurate "continuous" rate profile)
    traj['S'] = []
    traj['F'] = []
    traj['L'] = []
    for n in npoints:
        try:
            y = np.vstack([i[:,1] for i in data['S'] if len(i)==n]).mean(0)
            yerr = np.vstack([i[:,1] for i in data['S'] if len(i)==n]).std(0, ddof=1)
            traj['S'].append( (y, interp1d(range(n), y)(x_interp*(n-1)), yerr) )
        except ValueError:
            traj['S'].append( (np.zeros(n)+np.nan, np.zeros(x_interp.size)+np.nan, np.zeros(n)+np.nan) )

        try:
            y = np.vstack([i[:,1] for i in data['F'] if len(i)==n]).mean(0)
            yerr = np.vstack([i[:,1] for i in data['F'] if len(i)==n]).std(0, ddof=1)
            # account for endpoint bias on mean trajectories
            sbias = size_endpoint_bias(clusters, np.array(clusterix['F'])[[i for i,f in enumerate(data['F'])
                                                                           if len(f)==n]])
            y[0] -= sbias*y.sum()
            y[-1] -= sbias*y.sum()
            traj['F'].append( (y, interp1d(range(n), y)(x_interp*(n-1)), yerr) )
        except ValueError:
            traj['F'].append( (np.zeros(n)+np.nan, np.zeros(x_interp.size)+np.nan, np.zeros(n)+np.nan) )

        try:
            y = np.vstack([i[:,1] for i in data['L'] if len(i)==n]).mean(0)
            yerr = np.vstack([i[:,1] for i in data['L'] if len(i)==n]).std(0, ddof=1)
            traj['L'].append( (y, interp1d(range(n), y)(x_interp*(n-1)), yerr) )
        except ValueError:
            traj['L'].append( (np.zeros(n)+np.nan, np.zeros(x_interp.size)+np.nan, np.zeros(n)+np.nan) )

    return data, traj, clusterix


# ================ #
# Helper functions #
# ================ #
def regularize_sizes(listtraj, bins, npoints, min_size=3, min_dur=4):
    """Turn sizes trajectory into coarse-grained rate profiles.

    Parameters
    ----------
    listtraj : list of ndarray
    bins : list
    npoints : list
    min_size : int, 3
    min_dur : int, 4

    Returns
    -------
    list of ndarray
    list of ints
        Indices of trajectories that were kept. These correspond to the indices of the
        clusters.
    """

    assert bins[0]>=min_dur
    
    reglisttraj = []
    keepix = []

    for i,xy in enumerate(listtraj):
        if xy[-1,0]>=min_dur and xy[:,1].sum()>=min_size:
            # round down to min bin size
            binix = np.searchsorted(bins, xy[-1,0], 'right')-1
            if binix<0:
                binix = 0
            reglisttraj.append( coarse_grain_by_time(xy.astype(int), npoints[binix], 'sum').astype(int) )

            # endpoint bias
            reglisttraj[-1][0,1] -= 1
            reglisttraj[-1][-1,1] -= 1

            keepix.append(i)

    return reglisttraj, keepix

def regularize_fatalities(listtraj, bins, npoints, min_size=3, min_dur=4):
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

    assert bins[0]>=min_dur
    
    reglisttraj = []
    keepix = []

    for i,xy in enumerate(listtraj):
        if xy[-1,0]>=min_dur and xy[:,1].sum()>=min_size:
            # round down to min bin size
            binix = np.searchsorted(bins, xy[-1,0], 'right')-1
            if binix<0:
                binix = 0
            reglisttraj.append( coarse_grain_by_time(xy.astype(int), npoints[binix], 'sum').astype(int) )

            keepix.append(i)

    return reglisttraj, keepix

def regularize_diameters(listtraj, bins, npoints, min_dur=4, min_diameter=1e-6):
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

    assert bins[0]>=min_dur
    
    reglisttraj = []
    keepix = []

    for i,xy in enumerate(listtraj):
        if xy[-1,0]>=min_dur and xy[-1,1]>=min_diameter:
            # round down to min bin size
            binix = np.searchsorted(bins, xy[-1,0], 'right')-1
            if binix<0:
                binix = 0
            reglisttraj.append( coarse_grain_by_time(xy, npoints[binix], 'mean') )
            reglisttraj[-1][:,1] = np.maximum.accumulate(reglisttraj[-1][:,1])

            keepix.append(i)

    return reglisttraj, keepix

def coarse_grain_by_time(xy, nbins, method='sum'):
    """Coarse-grain trajectory by discretized units of time, adding up all event that
    happened within the same discrete time interval together.
    
    Parameters
    ----------
    xy : ndarray
    nbins : int
    method : str, 'sum'
        'sum', 'max', 'mean'

    Returns
    -------
    ndarray
    """

    bins = np.linspace(0, xy[-1,0], nbins+1)
    bins[-1] += 1  # stick last event into last bin
    ix = np.digitize(xy[:,0], bins)-1
    
    newtraj = np.zeros((nbins,2))
    newtraj[:,0] = np.arange(nbins)
    for i,ix_ in enumerate(ix):
        if method=='sum' or method=='mean':
            newtraj[ix_,1] += xy[i,1]
        elif method=='max':
            newtraj[ix_,1] = max(newtraj[ix_,1], xy[i,1])
        else:
            raise NotImplementedError

    if method=='max':
        newtraj[:,1] = np.maximum.accumulate(newtraj[:,1])
    elif method=='mean':
        for ix_ in np.unique(ix):
            newtraj[ix_,1] /= (ix==ix_).sum()

    return newtraj
