# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
import pickle
import numpy as np
import pandas as pd


# ================= #
# Helper functions. #
# ================= #
def default_dr(event_type='battle'):
    """Return default directory for event type.

    Parameters
    ----------
    event_type : str, 'battle'

    Returns
    -------
    str
    """
    
    if event_type=='battle':
        return 'geosplits/africa/battle/full_data_set/bin_agg'
    elif event_type=='civ_violence':
        return 'geosplits/africa/civ_violence/full_data_set/bin_agg'
    elif event_type=='riots':
        return 'geosplits/africa/riots/full_data_set/bin_agg'
    else: raise Exception("Unrecognized event type.")

def load_default_pickles(event_type='battle', gridno=0):
    """For shortening the preamble on most Jupyter notebooks.
    
    Parameters
    ----------
    event_type : str, 'battle'
        'battle', 'civ_violence', 'riots'

    Returns
    -------
    pd.DataFrame
        subdf
    dict of lists
        gridOfSplits
    """

    # voronoi binary aggregation
    region = 'africa'
    prefix = 'voronoi_noactor_'
    method = 'voronoi'
    folder = 'geosplits/%s/%s/full_data_set/bin_agg'%(region,event_type)

    # Load data
    subdf = pickle.load(open('%s/%sdf.p'%(folder, prefix),'rb'))['subdf']
    L = 9
    T = 11

    fname = '%s/%s%s'%(folder,prefix,'grid%s.p'%str(gridno).zfill(2))
    gridOfSplits = pickle.load(open(fname,'rb'))['gridOfSplits']
    
    return subdf, gridOfSplits

def rate_dynamics(dxdt=((160,16), (160,32), (160,64), (160,128), (160,256))):
    """Extract exponents from rate dynamical profiles.

    Automating analysis in "2019-11-18 rate dynamics.ipynb".

    Parameters
    ----------
    dxdt : tuple, ((160,16), (160,32), (160,64), (160,128), (160,256))

    Returns
    -------
    ndarray
        Dynamical exponent d_S/z.
    ndarray
        Dynamical exponent d_F/z.
    ndarray
        Dynamical exponent 1/z.
    """

    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist
    from .rate_dynamics import extract_from_df, interp_clusters

    dsz = np.zeros(len(dxdt))
    dfz = np.zeros(len(dxdt))
    invz = np.zeros(len(dxdt))
    subdf, gridOfSplits = load_default_pickles()

    for counter,dxdt_ in enumerate(dxdt):
        clusters = extract_from_df(subdf, gridOfSplits[dxdt_])

        # logarithmically spaced bins
        xinterp = np.linspace(0, 1, 100)
        bins = 2**np.arange(2, 14)
        npoints = np.arange(3, 20)[:bins.size-1]
        data, traj, clusterix = interp_clusters(clusters, xinterp, bins, npoints)


        offset = int(np.log2(dxdt_[1]))-2  # number of small bins to skip (skipping all avalanches with
                                           # duration <a days)

        # use variance in log space
        # these aren't weighted by number of data points because that returns something similar and is more
        # complicated
        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['S'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dsz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['F'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dfz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['L'][i+offset][1]/bins[i+offset]**a)
                
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        invz[counter] = minimize(cost, .5)['x']
        
        print("dxdt = ",dxdt_)
        print('ds/z = %1.2f'%dsz[counter])
        print('df/z = %1.2f'%dfz[counter])
        print('1/z = %1.2f'%invz[counter])
        print()
    
    return dsz, dfz, invz
