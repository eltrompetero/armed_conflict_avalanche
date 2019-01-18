# =============================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
from scipy.interpolate import interp1d
import multiprocess as mp
import numpy as np
import pickle


def count_up_unique_actors(x):
    ua = set()
    n = np.zeros(len(x),dtype=int)
    for i,x_ in enumerate(x):
        ua = set.union(ua,set(x_))
        n[i] = len(ua)
    return n

def add_same_date_events(t, x):
    uniqt = np.unique(t)
    xagg = np.zeros(len(uniqt))
    for i,t_ in enumerate(uniqt):
        xagg[i] = x[t==t_].sum()
    return uniqt, xagg

def avalanche_trajectory(g, min_len=5, min_size=5):
    """Extract from data the discrete sequence of events and their sizes.
    
    Parameters
    ----------
    g : pd.DataFrame
    min_len : int, 5
        Shortest duration of avalanche permitted for inclusion.
    min_size : int, 5
        Least number of unique events for inclusion.
        
    Returns
    -------
    ndarray
        List of (day, size) tuples.
    ndarray
        List of (day, fatalities) tuples.
    """
    
    dateFat, dateSize = [],[]
    durSize = []
    durFat = []
    for df in g:
        dur_ = (df.iloc[-1]['EVENT_DATE']-df.iloc[0]['EVENT_DATE']).days
        if dur_>=min_len:
            t = (df['EVENT_DATE']-df.iloc[0]['EVENT_DATE']).apply(lambda x:x.days).values
            
            t_, s = np.unique(t, return_counts=True)
            if s.sum()>=min_size:
                dateSize.append( np.vstack((t_, s)).T.astype(float) )
                durSize.append(dur_)
                
                if df['FATALITIES'].any():
                    f = df['FATALITIES'].values
                    t, f = add_same_date_events(t, f)
                    dateFat.append( np.vstack((t, f)).T.astype(float) )
                    durFat.append(dur_)

    return dateSize, dateFat, durSize, durFat

def interp_avalanche_trajectory(dateFat, x, insert_zero=True):
    """Average avalanche trajectory over many different avalanches using linear
    interpolation. Inserts 0 at the beginning and repeats max value at end. Since
    trajectories are normalized to 1, return the total size.

    Parameters
    ----------
    dateFat : pd.DataFrame
    x : ndarray
    insert_zero : bool, True
        If True, insert zero at beginning of time series to ensure that CDF starts at 0
        and add 1 at end.
    
    Returns
    -------
    ndarray
        Each row is an interpolated cumulative trajectory.
    """
    
    # traj of each avalanche in given data set
    traj = np.zeros((len(dateFat),len(x)))
    totalSize = np.zeros(len(dateFat))

    for i,df in enumerate(dateFat):
        totalSize[i] = df[:,1].sum()
        if insert_zero:
            # rescaled time
            x_ = np.append(np.insert((df[:,0]+1)/(df[-1,0]+2), 0, 0), 1)
            # cumulative profile
            y_ = np.append(np.insert(np.cumsum(df[:,1])/totalSize[i], 0, 0), 1)
        else:
            # rescaled time
            x_ = df[:,0]/df[-1,0]
            # cumulative profile
            y_ = np.cumsum(df[:,1])/totalSize[i]
        
        # assert not np.isnan(x_).any() and not np.isnan(y_).any()
        traj[i] = interp1d(x_, y_)(x)

    return traj, totalSize

def load_trajectories(event_type, dx, dt, gridno,
                      prefix='voronoi_noactor_',
                      region='africa',
                      n_interpolate=250):
    """Wrapper for interpolate size and fatalities trajectories from given file for given
    spatiotemporal scales.

    Parameters
    ----------
    event_type : str
        'battle', 'civ_violence', 'riots'
    dx : list
        Length scales.
    dt : list
        Time scales
    gridno : int
    prefix : str, 'voronoi_noactor_'
    region : str, 'africa'
    n_interpolate : int, 250

    Returns
    -------
    tuple
        Size profiles (traj, duration, total size). Each element organized by given
        cluster scales.
    tuple
        Fatality profiles (traj, duration, total size).
    """
    
    # load data and take all avalanches at each scale
    dr = 'geosplits/%s/%s/full_data_set/bin_agg'%(region, event_type)
    fname = '%s/%sgrid%s.p'%(dr, prefix, str(gridno).zfill(2))
    subdf = pickle.load(open('%s/%sdf.p'%(dr, prefix), 'rb'))['subdf']
    print("Loading %s"%fname)
    gridOfSplits = pickle.load(open(fname, 'rb'))['gridOfSplits']
    clustersix = [gridOfSplits[(dx_,dt_)] for dx_,dt_ in zip(dx, dt)]
    x = np.linspace(0,1,n_interpolate)
    
    sizeTrajByCluster = []
    fatTrajByCluster = []
    durSizeByCluster = []
    durFatByCluster = []
    totalSizeByCluster = []
    totalFatByCluster = []

    for i,c in enumerate(clustersix):
        # all subsets of subdf corresponding to clusters at this scale
        clusters = [subdf.loc[ix] for ix in c]

        # Get all raw sequences that are above some min length
        sizeTraj, fatTraj, durSize, durFat = avalanche_trajectory(clusters)
        durSizeByCluster.append(durSize)
        durFatByCluster.append(durFat)

        # interpolate trajectories
        traj, totalSize = interp_avalanche_trajectory(sizeTraj, x)
        assert len(traj)==len(totalSize)==len(durSize)
        sizeTrajByCluster.append( traj )
        totalSizeByCluster.append( totalSize)

        traj, totalFat = interp_avalanche_trajectory(fatTraj, x)
        assert len(traj)==len(totalFat)==len(durFat)
        fatTrajByCluster.append( traj )
        totalFatByCluster.append( totalFat )
    return ((sizeTrajByCluster, durSizeByCluster, totalSizeByCluster), 
            (fatTrajByCluster, durFatByCluster, totalFatByCluster)) 

def parallel_load_trajectories(event_type, gridno, dx, dt, **kwargs):
    """
    Loop over each file (for each random grid).
    
    Parameters
    ----------
    event_type : str
        'battle', 'civ_violence', 'riots'
    gridno : list
    dx : list
        Every pair (dx_,dt_) in zip(dx,dt) will be iterated.
    dt : list
    **kwargs
    
    Returns
    -------
    dict
        Sizes
    dict
        Fatalities 
    """
    
    def f(i):
        return load_trajectories(event_type, dx, dt, i, **kwargs)
    
    pool = mp.Pool(mp.cpu_count()-1)
    sizeInfo, fatInfo = zip(*pool.map(f, gridno))
    pool.close()
    
    sizeProfiles = {'traj':[], 'norm':[], 'dur':[]}
    fatProfiles = {'traj':[], 'norm':[], 'dur':[]}

    for i,g in enumerate(gridno):
        sizeProfiles['traj'].append(sizeInfo[i][0])
        sizeProfiles['dur'].append(sizeInfo[i][1])
        sizeProfiles['norm'].append(sizeInfo[i][2])

        fatProfiles['traj'].append(fatInfo[i][0])
        fatProfiles['dur'].append(fatInfo[i][1])
        fatProfiles['norm'].append(fatInfo[i][2])

    return sizeProfiles, fatProfiles

def average_trajectories_by_coarseness(trajectories):
    """Average over the trajectories calculated from various random grids to get error
    bars. Since each trajectory is not normalized, must normalize by length.
    
    Parameters
    ----------
    trajectories : list
        Where outermost list is by gridno (typically 10 of these).
        
    Returns
    -------
    list
        List of tuples containing the averaged trajectory and the std. Outmost length is
        the number of clusters given.
    """
    
    trajByCoarseness = list(zip(*trajectories))
    avgTrajAndError = []
    
    for y in trajByCoarseness:
        y = np.vstack([i.mean(0) for i in y])
        # average taken over random grids
        avgTrajAndError.append( (y.mean(0), y.std(axis=0,ddof=1)) )
    return avgTrajAndError

def avalanche_trajectory_ua(g, x, min_len=5):
    raise NotImplentedError("Needs updating.")
    dateFat=[]
    for ev in g:
        df = pd.concat(ev)
        if len(df)>=min_len:
            t = (df['EVENT_DATE']-df.iloc[0]['EVENT_DATE']).apply(lambda x:x.days).values
            
            if (t>0).any() and df['FATALITIES'].any():
                uaCount = count_up_unique_actors( df['actors'].values )
                t, uaCount = add_same_date_events(t, uaCount)
                
                dateFat.append( np.vstack((t,uaCount)).T.astype(float) )
    
    avgTrajectory = np.zeros_like(x)
    for df in dateFat:
        x_=df[:,0]/df[-1,0]
        y_=np.cumsum(df[:,1]/df[:,1].sum())
        assert not np.isnan(x_).any() and not np.isnan(y_).any()
        
        avgTrajectory += interp1d(x_, y_)(x)
    avgTrajectory /= len(dateFat)
    
    return avgTrajectory, dateFat
