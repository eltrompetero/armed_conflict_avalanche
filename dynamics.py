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

def avalanche_trajectory(g, min_len=4, min_size=2, min_fat=2):
    """Extract from data the discrete sequence of events and their sizes.
    
    Parameters
    ----------
    g : pd.DataFrame
    min_len : int, 4
        Shortest duration of avalanche permitted for inclusion.
    min_size : int, 2
        Least number of unique events for inclusion.
    min_fat : int, 2
        
    Returns
    -------
    ndarray
        List of (day, size) tuples.
    ndarray
        List of (day, fatalities) tuples.
    """
    
    dateFat, dateSize, durSize, durFat = [],[],[],[]
    for df in g:
        dur_ = (df.index[-1] - df.index[0]).days
        if dur_>=min_len:
            t = np.array((df.index-df.index[0]).days.tolist())
            s = df['SIZES'].values 
            if s.sum()>=min_size:
                dateSize.append( np.vstack((t, s)).T.astype(float) )
                durSize.append(dur_)
                
                # fatalities must be spread out over at least two different days
                if df['FATALITIES'].sum()>=min_fat & (df['FATALITIES'].values>0).sum()>1:
                    f = df['FATALITIES'].values
                    dateFat.append( np.vstack((t, f, s)).T.astype(float) )
                    durFat.append(dur_)

    return dateSize, dateFat, np.array(durSize), np.array(durFat)

def interp_avalanche_trajectory(dateFat, x,
                                cum=True,
                                insert_zero=False,
                                append_one=False,
                                symmetrize=False,
                                run_checks=False):
    """Average avalanche trajectory over many different avalanches using linear
    interpolation. Can insert 0 at the beginning and repeat max value at end. Since
    trajectories are normalized to 1, return the total size.

    Since by defn, CDFs must end at 1, it make sense to likewise fix the beginning to
    start at 0 to see how the curves behave in the limit of larger conflicts.

    Parameters
    ----------
    dateFat : pd.DataFrame
    x : ndarray
    cum : bool, True
        If True, return cumulative form interpolated.
    insert_zero : bool, False
        If True, insert zero at beginning of time series to ensure that CDF starts at 0.
    append_one : bool, False
        Add 1 at end.
    symmetrize : bool, False
    
    Returns
    -------
    ndarray
        Each row is an interpolated cumulative trajectory.
    """
    
    # traj of each avalanche in given data set
    traj = np.zeros((len(dateFat),len(x)))
    totalSize = np.zeros(len(dateFat))
    
    if cum:
        for i,df in enumerate(dateFat):
            totalSize[i] = df[:,1].sum()
            if insert_zero and append_one:
                # rescaled time
                x_ = np.append(np.insert((df[:,0]+1)/(df[-1,0]+2), 0, 0), 1)
                # cumulative profile
                y_ = np.append(np.insert(np.cumsum(df[:,1])/totalSize[i], 0, 0), 1)
            elif insert_zero:
                # rescaled time
                x_ = np.insert((df[:,0]+1)/(df[-1,0]+1), 0, 0)
                # cumulative profile
                y_ = np.insert(np.cumsum(df[:,1])/totalSize[i], 0, 0)
            elif append_one:
                # rescaled time
                x_ = np.append(df[:,0]/(df[-1,0]+1), 1)
                # cumulative profile
                y_ = np.append(np.cumsum(df[:,1])/totalSize[i], 1)
            elif symmetrize:
                # rescaled time
                x_ = df[:,0]/df[-1,0]
                # cumulative profile
                y_ = np.cumsum(df[:,1])/totalSize[i]
                # symmetrize cdf
                y_ = (y_+np.insert(y_[:-1],0,0))/2
                # account for bias on end points
                y_[0] *= 2
                y_[-1] = y_[-1]*2-1

                if run_checks:
                    assert y_[0]<y_[1] or abs(y_[0]-y_[1])<1e-12, (y_[0],y_[1])
                    assert y_[-1]>y_[-2] or abs(y_[-1]-y_[-2])<1e-12, (y_[-1],y_[-2])
            else:
                # rescaled time
                x_ = df[:,0]/df[-1,0]
                # cumulative profile
                y_ = np.cumsum(df[:,1])/totalSize[i]
            
            traj[i] = interp1d(x_, y_)(x)
    
    else:
        print("Rate profile...")
        for i,df in enumerate(dateFat):
            totalSize[i] = df[:,1].sum()
            # rescaled time
            x_ = (np.linspace(0,df[-1,0],6)[:-1] + np.linspace(0,df[-1,0],6)[1:])/2/df[-1,0]
            x_ = np.append(np.insert(x_,0,0), 1)
            # rate profile
            y_ = np.zeros(7)
            y_[0] = df[0,1]
            y_[-1] = df[-1,1]
            timebinix = np.digitize(df[1:-1,0], np.linspace(0,df[-1,0],6))
            df = df[1:-1]
            for j in range(1,6):
                y_[j] = df[timebinix==j,1].sum()
            y_ = y_/totalSize[i]
            traj[i] = interp1d(x_, y_)(x)

            if run_checks: 
                assert (x_.max()<=1)&(x_.min()>=0)&(np.diff(x)>0).all()
                assert (y_.max()<=1)&(y_.min()>=0), y_

    return traj, totalSize

def load_trajectories(event_type, dx, dt, gridno,
                      prefix='voronoi_noactor_',
                      region='africa',
                      n_interpolate=250,
                      shuffle=False,
                      only_rate=False,
                      reverse=False,
                      smear=False,
                      cum=False):
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
    shuffle : bool, False
        If True, shuffle the sizes and fatalities to get a "time shuffled" version of the
        trajectories.
    only_rate : bool, False
        If True, return cum profile of avalanches rate.
    reverse : bool, False
        If True, reverse time order of avalanche events.

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
    subdf['SIZES'] = np.ones(len(subdf))
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
        clusters = [subdf.loc[ix,('EVENT_DATE','FATALITIES','SIZES')] for ix in c]
        printed = False 
        # reorganize by unique event per day
        for i,c in enumerate(clusters):
            clusters[i] = c.groupby('EVENT_DATE').sum() 
            if shuffle:
                randix = np.random.permutation(len(clusters[i]))
                clusters[i]['FATALITIES'] = clusters[i]['FATALITIES'].values[randix]
                clusters[i]['SIZES'] = clusters[i]['SIZES'].values[randix]
            elif only_rate:
                clusters[i]['FATALITIES'] = np.clip(clusters[i]['FATALITIES'].values, 0, 1)
                clusters[i]['SIZES'] = np.ones(len(clusters[i]))
            elif reverse:
                clusters[i].index = (clusters[i].index[-1] -
                                     (clusters[i].index-clusters[i].index[0]))
                clusters[i] = clusters[i].iloc[::-1]

            elif smear:
                from datetime import timedelta
                
                if len(clusters[i])>2:
                    #t0 = (clusters[i].index[-1]-clusters[i].index[0]).days
                    newdays = np.zeros(len(clusters[i]), dtype=int)
                    # pick random days to be first and last
                    firstandlast = np.random.choice(range(len(clusters[i])), size=2, replace=False)
                    newdays[firstandlast[1]] = (clusters[i].index[-1]-clusters[i].index[0]).days
                    # randomly fill in between
                    remainingix = np.delete(range(len(clusters[i])), firstandlast)
                    newdays[remainingix] = np.random.choice(range(1,(clusters[i].index[-1]-clusters[i].index[0]).days),
                                                            size=len(clusters[i])-2, replace=False)
                    #assert np.unique(newdays).size==len(clusters[i])

                    # make new dataframe
                    newix = []
                    for dix,d in enumerate(newdays):
                        newix.append( clusters[i].index[0] + timedelta(days=int(d)) )
                    clusters[i].index = newix
                    clusters[i] = clusters[i].sort_index()
                    #assert t0==(clusters[i].index[-1]-clusters[i].index[0]).days
                    #print(clusters[i].index)
                    #assert (np.diff(clusters[i].index).astype(int)>=0).all()

        # Get all raw sequences that are above some min length
        sizeTraj, fatTraj, durSize, durFat = avalanche_trajectory(clusters)

        durSizeByCluster.append(durSize)
        durFatByCluster.append(durFat)

        # interpolate trajectories
        traj, totalSize = interp_avalanche_trajectory(sizeTraj, x, cum=cum)
        assert len(traj)==len(totalSize)==len(durSize)
        sizeTrajByCluster.append( traj )
        totalSizeByCluster.append( totalSize)

        traj, totalFat = interp_avalanche_trajectory(fatTraj, x, cum=cum)
        assert len(traj)==len(totalFat)==len(durFat)
        fatTrajByCluster.append( traj )
        totalFatByCluster.append( (totalFat,[i[:,2].sum() for i in fatTraj]) )
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
