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

def avalanche_trajectory(g, min_len=5):
    """Extract from data the discrete sequence of events and their sizes.
    
    Parameters
    ----------
    g : pd.DataFrame
    min_len : int, 5
        Shortest duration of avalanche permitted for inclusion.
        
    Returns
    -------
    ndarray
        List of (day, size) tuples.
    ndarray
        List of (day, fatalities) tuples.
    """
    
    dateFat, dateSize = [],[]
    for df in g:
        if (df.iloc[-1]['EVENT_DATE']-df.iloc[0]['EVENT_DATE']).days>=min_len:
            t=(df['EVENT_DATE']-df.iloc[0]['EVENT_DATE']).apply(lambda x:x.days).values
            
            s = np.ones(len(t))
            t_, s = add_same_date_events(t, s)
            dateSize.append( np.vstack((t_, s)).T.astype(float) )
            
            if df['FATALITIES'].any():
                f = df['FATALITIES'].values
                t, f = add_same_date_events(t, f)
                dateFat.append( np.vstack((t, f)).T.astype(float) )
    return dateSize, dateFat

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

def interp_avalanche_trajectory(dateFat, x):
    """Average avalanche trajectory over many different avalanches using linear
    interpolation. Inserts 0 at the beginning.
    
    Returns
    -------
    ndarray
        Avg
    ndarray
        Std
    list
    """
    
    avgTrajectory = np.zeros((len(dateFat),len(x)))
    for i,df in enumerate(dateFat):
        # rescaled time (0,1)
        x_ = np.insert((df[:,0]+1)/(df[-1,0]+1), 0, 0)
        # cumulative profile
        y_ = np.insert(np.cumsum(df[:,1]/df[:,1].sum()), 0, 0)
        # assert not np.isnan(x_).any() and not np.isnan(y_).any()
        avgTrajectory[i] = interp1d(x_,y_)(x)

        # number of events per day
#         y_=df[:,1]
#         # assert not np.isnan(x_).any() and not np.isnan(y_).any()
#         avgTrajectory[i] = interp1d(x_,y_)(x)
#         avgTrajectory[i] /= trapz(avgTrajectory[i], x=x)
        
    return (avgTrajectory.mean(0),
            avgTrajectory.std(axis=0, ddof=1)/np.sqrt(len(avgTrajectory)),
            dateFat)

def _trajectories(dx, dt, i, dr, prefix, x=np.linspace(0,1,1000)):
    """Wrapper for interpolate size and fatalities trajectories from given file for given
    spatiotemporal scales.

    Parameters
    ----------
    dx : list
        Length scales.
    dt : list
        Time scales
    dr : str
    prefix : str
    x : ndarray

    Returns
    -------
    list
    list
    int
    """
    
    # load data
    fname = '%s/%sgrid%s.p'%(dr, prefix, str(i).zfill(2))
    subdf = pickle.load(open('%s/%sdf.p'%(dr, prefix), 'rb'))['subdf']
    print("Loading %s"%fname)
    gridOfSplits = pickle.load(open(fname, 'rb'))['gridOfSplits']
    clustersix = [gridOfSplits[(dx_,dt_)] for dx_,dt_ in zip(dx, dt)]
    nDataPoints = np.zeros((len(clustersix),2))
    
    # interpolate
    interpTrajectories = []
    for i,c in enumerate(clustersix):
        clusters = [subdf.loc[ix] for ix in c]
        # Get all avalanche trajectories in these events that are above some min length
        sizeTraj, fatTraj = avalanche_trajectory(clusters)
        
        interpTrajectories.append((interp_avalanche_trajectory(sizeTraj, x)[:2],
                                   interp_avalanche_trajectory(fatTraj, x)[:2]))
        nDataPoints[i] = len(sizeTraj), len(fatTraj)
    interpSizeTraj, interpFatTraj = list(zip(*interpTrajectories))
    return interpSizeTraj, interpFatTraj, nDataPoints

def load_trajectories(dr, nfiles, dx, dt, prefix=''):
    """Load the data sets, extract the long avalanches, and interpolate trajectories. 
    Loop over each file (for each random grid).
    
    Parameters
    ----------
    dr : str
    nfiles : int
        Count up to this number assuming that files are named as _00, _01, ...
    dx : list
        Every pair (dx_,dt_) in zip(dx,dt) will be iterated.
    dt : list
    prefix : str, ''
    
    Returns
    -------
    list
        Fatalities 
    list
        Sizes
    list of ndarray
        Number of data points. Columns are (size, fatalities)
    """
    
    def f(i):
        return _trajectories(dx, dt, i, dr, prefix)
    
    pool = mp.Pool(mp.cpu_count()-1)
    sTrajByFile, fTrajByFile, nDataPoints = zip(*pool.map(f, range(nfiles)))
    pool.close()
    
    return sTrajByFile, fTrajByFile, nDataPoints

def average_trajectories_by_coarseness(trajectories):
    """Average over the trajectories calculated from various random grids to get error bars.
    
    Parameters
    ----------
    trajectories : list
    
    Returns
    -------
    """
    
    trajByCoarseness=list(zip(*trajectories))
    avgTrajAndError=[]
    
    for t in trajByCoarseness:
        i = np.vstack(t)
        # average taken over random grids
        avgTrajAndError.append( (i.mean(0), i.std(axis=0,ddof=1)) )
    return avgTrajAndError
