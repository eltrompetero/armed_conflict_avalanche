import numpy as np
from sklearn.cluster import MeanShift
from sklearn.neighbors import BallTree,DistanceMetric
from scipy.spatial.distance import pdist
import pandas as pd
import os
from datetime import datetime
from geopy.distance import vincenty,great_circle
from itertools import combinations,chain
from scipy.spatial.distance import squareform
from misc.plot import colorcycle
from misc.utils import unique_rows
import scipy.stats as stats
from scipy.optimize import minimize
import gmplot
from warnings import warn
from misc.globe import SphereCoordinate,PoissonDiscSphere
from numba import jit,njit
DATADR = os.path.expanduser('~')+'/Dropbox/Research/armed_conflict/data/'


def coarse_grain_voronoi_tess(dx, fileno):
    """
    Successive coarse-graining from lowest layer to each upper layers. To keep the coarse-graining
    consistent, coarse-graining happens only between adjacent levels and then the entire
    coarse-graining operation is traced from the bottom to the top layer.

    Parameters
    ----------
    dx : list
        Spacing of adjacent layers with which to coarse grain.
    fileno : int
        Only file with this name will be taken from all layers specified in dx.
        
    Returns
    -------
    nextLayerPixel : list
        Each list maps lowest (finest) layer to each of the upper layers.
    """
    import pickle
    assert (np.diff(dx)<0).all(), "Grid must get coarser."
    
    # mappings between adjacent layers
    nextLayerPixel=_coarse_grain_voronoi_tess(dx, fileno)
    
    # Iterate through layers and track to back bottom-most layer
    bottomLayerMapping=[nextLayerPixel[0]]
    for el in range(1, len(dx)-1):
        bottomLayerMapping.append(nextLayerPixel[el][bottomLayerMapping[el-1]])
    return bottomLayerMapping

def _coarse_grain_voronoi_tess(dx, fileno):
    """
    Coarse-graining by adjacent layers.

    Parameters
    ----------
    dx : list
        Spacing of adjacent layers with which to coarse grain.
    fileno : int
        Only file with this name will be taken from all layers specified in dx.
        
    Returns
    -------
    nextLayerPixel : list
        Each list maps dx[i] to dx[i+1] by pixel index in dx[i+1], i.e. each entry names the
        coarse-grained pixel to which it belongs such that the length of this array is as long as
        the number of pixels in the fine-grained layer.
    """
    import pickle
    assert (np.diff(dx)<0).all(), "Grid must get coarser."
    
    poissd=[]  # hold two adjacent Poisson disc samples
    nextLayerPixel=[]  # hold the mappings
    
    # Iterate through layers
    poissd.append(pickle.load(open('voronoi_grids/%d/%s.p'%(dx[0],str(fileno).zfill(2)),
                                   'rb'))['poissd'])
    for el in range(len(dx)-1):
        # load the next coarse grained layer
        poissd.append(pickle.load(open('voronoi_grids/%d/%s.p'%(dx[el+1],str(fileno).zfill(2)),
                                       'rb'))['poissd'])
        
        nextLayerPixel.append(np.zeros(len(poissd[0].samples), dtype=int))
        for ptix,pt in enumerate(poissd[0].samples):
            nextLayerPixel[-1][ptix]=poissd[1].get_closest_neighbor(pt)[0]
            
        # remove the finest grain layer so we can load the next layer
        poissd.pop(0)
            
    return nextLayerPixel

def voronoi_pix_diameter(spaceThreshold):
    pixDiameter=np.zeros(len(spaceThreshold))
    for i,dx in enumerate(spaceThreshold):
        pixDiameter[i]=_sample_lattice_spacing(dx, 10)
    return pixDiameter

def _sample_lattice_spacing(dx, sample_size):
    """
    Parameters
    ----------
    dx : float
    sample_size : int
        no of random points to use to estimate radius
        
    Returns
    -------
    dist : float
        In units of km on Earth.
    """
    import pickle
    poissd=pickle.load(open('voronoi_grids/%d/00.p'%dx,'rb'))['poissd']
    
    if sample_size>len(poissd.samples):
        sample_size=len(poissd.samples)
        
    randix=np.random.choice(np.arange(len(poissd.samples)), size=sample_size, replace=False)

    d=np.zeros(sample_size)
    for ix in randix:
        d=poissd.get_closest_neighbor_dist(poissd.samples[ix])

    return d.mean()*6370

def sample_sphere(n=1,degree=True):
    if degree:
        randlat=np.arccos(2*np.random.rand(n)-1)/np.pi*180-90
        randlon=np.random.uniform(-180,180,size=n)
        return randlon,randlat
    randlat=np.arccos(2*np.random.rand(n)-1)-np.pi/2
    randlon=np.random.uniform(-np.pi,np.pi,size=n)
    return randlon,randlat

def loglog_fit(x, y, p=2, iprint=False, full_output=False):
    """Symmetric log-log fit."""
    from scipy.optimize import minimize
    def cost(params):
        a,b=params
        return (np.abs(a*np.log(x)+b-np.log(y))**p).sum()+(np.abs(np.log(x)+b/a-np.log(y)/a)**p).sum()
    soln=minimize(cost, [1,0])
    if iprint and not soln['success']:
        print("loglog_fit did not converge on a solution.")
        print(soln['message'])
    if full_output:
        return soln['x'], soln
    return soln['x']

def loglog_fit_err_bars(x, y, fit_params):
    """Assuming Gaussian error bars on the loglog_fit, use Laplace approximation to get error bars
    that correspond to the distance spanned by the covariance matrix in each dimension.

    The covariance matrix is calculated from the inverse Hessian.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    fit_params : twople

    Returns
    -------
    std : ndarray
        Standard deviation for (da,db) from log(y)=a * log(x) + b.
    """

    import numdifftools as ndt

    #f=lambda args:((np.log(y) - args[0]*np.log(x)-args[1])**2 + 
    #               (np.log(y)/args[0] - np.log(x)-args[1]/args[0])**2).sum()/2
    #hess=ndt.Hessian(f)(fit_params)*2  # factor of 2 from taylor expansion
    #cov=np.linalg.inv(hess)
    # project max extension along parameter a axis as if variances were additive
    #eigval, eigvec=np.linalg.eig(cov)
    #return np.sqrt( eigval.dot(np.abs(eigvec.T)) )
    f=lambda args:np.concatenate((np.log(y) - args[0]*np.log(x)-args[1],
                                  np.log(y)/args[0] - np.log(x)-args[1]/args[0]))
    # return standard error of the mean from log likelihood estimation of parameters
    return f(fit_params).std()/np.sqrt(len(x)),0

def extract_ua_from_geosplit(geoSplit):
    """Pull out sets of unique actors from each avalanche listed in geoSplit.

    Parameters
    ----------
    geoSplit : list of avalanches

    Returns
    -------
    unique_actors : list of ndarray
    """
    ua=[]
    for i,g in enumerate(geoSplit):
        # NOTE:Handle special exception for buggy version of previous code. This should be removed
        # once new version of tpixelate has been used.
        if not type(g) is list:
            g=[g]

        try:
            df=pd.concat(g)
            ua.append( np.unique( np.concatenate( df.loc[:,'actors'].values ) ) )
        except TypeError:
            print(i,g)
            raise Exception
    return ua

def cluster_diameter(lonlat):
    """Return largest distance (km) between all pairs of latlon coordinates.
    
    Parameters
    ----------
    lonlat : ndarray
    
    Returns
    -------
    dmax : float
    """
    from misc.globe import haversine

    if len(lonlat)>1:
        counter=0
        d=pdist(lonlat/180*np.pi, lambda u,v: haversine(u,v,6370))
        return d.max()
    return 0

def duration_and_size(geoSplit):
    """Extract interesting info from clustered data."""
    eventCount=[]
    duration=[]
    fatalities=[]
    diameter=[]
    
    for av in geoSplit:
        if type(av) is pd.DataFrame:
            av=[av]
    
        eventCount.append(sum([len(i) for i in av]))
        duration.append(len(av))

        f=0
        for i in av:
            f+=i['FATALITIES'].values.astype(int).sum()
        fatalities.append(f)
        
        if len(av)==1:
            diameter.append(0.)
        else:
            diameter.append(cluster_diameter(pd.concat(av).loc[:,('LONGITUDE','LATITUDE')].values))
    
    return eventCount, duration, fatalities, diameter

def split_by_nan(v):
    """
    Split a vector into non-nan segments that are separated by nans.
    
    Parameters
    ----------
    v : ndarray
    
    Returns
    -------
    v_separated : list
        Each element is a vector of a contiguous sequence of non-nan numbers.
    ix : list
        Each element is the index of the elements that correspond to the given elements in the first variable.
    """
    assert len(v)>1
    
    if np.isnan(v).all():
        return [],[]
    if (np.isnan(v)==0).all():
        return [v],[np.arange(len(v))]
    
    ix=np.arange(len(v),dtype=float)
    ix[np.isnan(v)]=np.nan
    
    b=[i for i in np.split(ix,np.where(np.isnan(ix))[0]) if len(i)>1 or (len(i)==1 and np.isnan(i[0])==0)]
    b=[i.astype(int) if np.isnan(i[0])==0 else i[1:].astype(int) for i in b]
    return [v[i] for i in b],b

def fast_cluster_diameter(lonlat, n_top=3):
    """Return largest distance (km) between all pairs of lonlat coordinates.
    
    In this sped up version, first look for simple distance using Euclidean distance with
    coordinates and then convert the top few results to km.

    Parameters
    ----------
    lonlat : ndarray
    n_top : int,3

    Returns
    -------
    d_max : float
        Furthest distance between two points in the list.
    """
    from misc.utils import unravel_utri
    from misc.globe import haversine
    
    if 1<len(lonlat)<=n_top:
        d=pdist(lonlat/180*np.pi, lambda u,v: haversine(u,v,6370))
        return d.max()
    else:
        d=pdist(lonlat)
        topix=np.argsort(d)[-n_top:]
        i,j=unravel_utri(topix, len(lonlat))
        uniqix=list(set(i.tolist()+j.tolist()))
        
        d=pdist(lonlat[uniqix]/180*np.pi, lambda u,v: haversine(u,v,6370))
        return d.max()
    return 0.

def euclid_dist(xxx_todo_changeme, xxx_todo_changeme1):
    (lat, lng) = xxx_todo_changeme
    (lat0, lng0) = xxx_todo_changeme1
    deglen = 110.25
    x = lat - lat0
    y = (lng - lng0)*np.cos(lat0)
    return deglen*np.sqrt(x*x + y*y)

def split_by_index(ix,X):
    """
    Split given dataframe at specified indices.
    """
    split = []
    if ix[0]>0:
        split.append(X.iloc[:ix[0]])
    for i in range(1,len(ix)-1):
        split.append(X.iloc[ix[i]:ix[i+1]])
    if ix[-1]<(len(X)-1):
        split.append(X.iloc[ix[-1]:])
    else:
        split.append(X.iloc[-1:])
    return split

def split_by_date(df,threshold=14,return_details=False):
    """Split dataframe where the number of days separating sequential events exceeds the
    threshold.

    Parameters
    ----------
    df : pandas.DataFrame
    threshold : int,14
    return_details : bool,False

    Returns
    -------
    splitdf : list of pd.DataFrame
    dt : list
        Time difference in days at the split.
    pairSizes : list
        Twoples of event sizes straddling the split.
    """
    try:
        # Handle case where the object type is pandas.Timedelta.
        dt = np.array([i.days for i in np.diff(df['EVENT_DATE'])])
    except AttributeError:
        # Handle case where the object is of type numpy.
        dt = np.array(np.diff(df['EVENT_DATE']),dtype=np.timedelta64)/np.timedelta64(1,'D')
    except:
        print(np.diff(df['EVENT_DATE']))
        raise Exception
    
    if (dt<0).any():
        warn("This DataFrame is not ordered in time.")

    ix = np.where(dt>threshold)[0]+1
    if len(ix)>0:
        splitdf=split_by_index(ix,df)
        if return_details:
            dt,pairSizes=zip(*[((splitdf[i].iloc[-1]['EVENT_DATE']-splitdf[i-1].iloc[-1]['EVENT_DATE']).days,
                                (len(splitdf[i]),len(splitdf[i-1])))
                               for i in range(1,len(splitdf))])
            return splitdf,dt,pairSizes
        return splitdf
    else:
        if return_details:
            return [df],[],[]
        return [df]

def cluster_dist(dmat,threshold):
    """Cluster using distance matrix."""
    clusters = []
    for i in range(len(dmat)):
        clusters.append(np.where(dmat[i]<threshold)[0].tolist())
    uclusters = []
    for c in clusters:
        if not c in uclusters:
            uclusters.append(c)
    return uclusters

def split_by_dist(df,threshold=2,fast=False):
    """Split data frame into list of pieces that are within a threshold distance of each other."""
    dmat = np.zeros((len(df),len(df)))
    latlon = np.vstack((df['LATITUDE'],df['LONGITUDE'])).T
    if fast:
        for i,j in combinations(list(range(len(dmat))),2):
            dmat[i,j] = euclid_dist(latlon[i],latlon[j])
    else:
        for i,j in combinations(list(range(len(dmat))),2):
            dmat[i,j] = great_circle(latlon[i],latlon[j]).meters/1000
    dmat += dmat.T
    
    split = []
    clustix = cluster_dist(dmat,threshold)
    for ix in clustix:
        split.append(df.iloc[ix])
    return split

def pipeline_civ_violence(df):
    """Extract violence against civilians events."""
    subdf = df.iloc[(df['EVENT_TYPE']=='Violence against civilians').values]

    split = []
    for a in unique(subdf['ACTOR1']):
        split.append(subdf.iloc[(subdf['ACTOR1']==a).values])
        split[-1].sort_values('EVENT_DATE',inplace=True)

    split_ = split

    # Split by date.
    split = [split_by_date(s) if (len(s)>1) else [s] for s in split]
    split = list(chain.from_iterable(split))

    # Split by distance
    split = [split_by_dist(s) if (len(s)>1) else [s] for s in split]
    split = list(chain.from_iterable(split))
    
    return split

def merge(sets):
    """Merge a list of sets such that any two sets with any intersection are merged."""
    merged = 1  # has a merge occurred?
    while merged:
        merged = 0
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = 1
                    common |= x
            results.append(common)
        sets = results
    return sets

def split_by_actor(subdf):
    """Split by participants.

    Parameters
    ----------
    subdf : pandas.DataFrame
        Should have actors column.

    Returns
    -------
    splitByActors : list of pandas.DataFrame
    uactors : list of sets
        Unique actors used to split the dataframe.
    """
    # Remove all rows that have null values. for actors
    subdf=subdf.iloc[ subdf['actors'].isnull().values==0 ]
    
    # Collect all sets of actors.
    uactors = []
    for a in subdf['actors']:
        if not a in uactors:
            uactors.append(a)
    
    # Combine all sets that intersect. Repeat til there are no intersections left. This should get a
    # connected chain of actors.
    uactors = [set(a) for a in uactors]
    # if there is only one event there can only be one unique set of actors
    if len(uactors)==1:
        return [subdf],uactors
    uactors=merge(uactors)
    
    # Split the DataFrame by these unique groups that were identified.
    split = [[] for i in range(len(uactors))]
    #for ai,a in subdf['actors'].iteritems():
    for ai,a in enumerate(subdf['actors']):
        for si,s in enumerate(uactors):
            if set(a)<=s:
                split[si].append(subdf.iloc[ai])
                break
    split = [pd.concat(s,axis=1,ignore_index=False).transpose() for s in split]
    #if not type(split) is list:
    #    split=[split]
    return split,uactors

def get_latlon(df):
    return np.vstack((df['LATITUDE'],df['LONGITUDE'])).T

def cluster_by_latlon(latlon,mxdist):
    """Use BallTree to cluster events by latitude longitude having used the haversine (great circle)
    distance with radius of the Earth given by 6.378km.
    
    This only keeps the unique clusters found so it is possible that some events repeat or that some
    are missing.
    """
    latlon = latlon*np.pi/180  # convert to radians
    tree = BallTree(latlon,leaf_size=20,metric='haversine')
    
    clusteredEventsIx = []
    for ll in latlon:
        # d is returned normalized by the radius of the earth
        d,ix = tree.query([ll],k=min([len(latlon),100]))
        ix = ix.ravel()[d.ravel()<(mxdist/6.378e3)]

        # If the number of neighbors exceeds the cutoff, then we need to expand our ball to include
        # more neighbors.
        cutoff = len(d[0])
        while len(ix)==cutoff and cutoff!=len(latlon):
            cutoff = min([cutoff*2,len(subdf)])
            d,ix = tree.query([ll],k=cutoff)
            ix = ix.ravel()[d.ravel()<(mxdist/6.378e3)]
        
        clusteredEventsIx.append(np.sort(ix.tolist()))
    
    # Only keep unique clusters.
    uClusteredEventsIx = [list(x) for x in set(tuple(x) for x in clusteredEventsIx)]
    return uClusteredEventsIx

def split_by_dist_fast(subdf,mxdist=40,method='manual'):
    """Instead of using the fully accurate geodesic distance, just use the haversine approximation.

    Parameters
    ----------
    subdf : pandas.DataFrame
        With columns LATITUDE AND LONGITUDE
    mxdist : float,40
        Length scale to use for clustering.
    method : str,'manual'
        If 'manual', use your own way of calling BallTree. If 'cluster', use the mean shift
        algorithm implemented in scikit-learn. That also uses BallTree to cluster events.
    """
    latlon = get_latlon(subdf)
    
    if method=='manual':
        uClusteredEventsIx = cluster_by_latlon(latlon,mxdist)
    elif method=='cluster':
        ms = MeanShift(bandwidth=mxdist/6.378e3,metric='haversine')
        ms.fit(latlon*np.pi/180)
        uClusteredEventsIx = []
        for i in np.unique(ms.labels_):
            uClusteredEventsIx.append( np.where(ms.labels_==i)[0] )
    else:
        raise Exception("Invalid method.")
    
    splitdf = []
    for c in uClusteredEventsIx:
        splitdf.append(subdf.iloc[c])
    return splitdf

def center(X,ddof=0):
    m = X.mean()
    width = X.std(ddof=ddof)
    
    return (X-m)/width,m,width

def square_fit_mean_std(m,s,return_all=False):
    """
    Fit a square root curve for the standard deviation given the mean with the
    assumption that these should be related to each other by a scaling
    constant.

    Params:
    -------
    m (array-like)
        Means.
    s (array-like)
        Standard deviations.

    Returns:
    --------
    Constant factor.
    """
    def cost(a):
        return ((a*np.sqrt(m)-s)**2).sum()
    if return_all:
        return minimize(cost,1)
    else:
        return minimize(cost,1)['x']

def square_val_mean_std(sqfit,m):
    return np.sqrt(m)*sqfit

def square_fit(x,y,weights=None):
    """
    Fit a function of the form a*sqrt(x)+b with max likelihood and option to
    provide weights on data points. Usually the weight is proportional to the
    number of data points.

    Params:
    -------
    x,y (array-like)
    weights (array-like=None)

    Returns:
    --------
    sqfit
        (a,b)
    """
    if weights is None:
        weights = np.ones_like(y)

    def cost(params):
        a,b = params
        return (( (a*np.sqrt(x)+b-y)**2 )*weights).sum()
    return minimize(cost,[0,0])['x']

def square_val(sqfit,x):
    return sqfit[0]*np.sqrt(x) + sqfit[1]

def intra_event_dt(splitdf,pairwise=False):
    """
    The number of days between sequential events of the sequence of events given or all pairs of events.
    
    Params:
    -------
    splitdf (list of DataFrames)
    pairwise (bool=False)
        If True, find the absolute value of the time differences between all pairs of events.

    Returns:
    --------
    intraEventdt (ndarray)
    """
    intraEventdt = []
    if pairwise:
        d = lambda u,v: (v[3]-u[3])/np.timedelta64(1,'D')
        for s in splitdf:
            intraEventdt.append( pdist(s,d) )
    else:
        # Assuming the sequence has been ordered.
        for s in splitdf:
            intraEventdt.append( np.diff(s['EVENT_DATE'])/np.timedelta64(1,'D') )

    intraEventdt = np.concatenate(intraEventdt).astype(int)
    intraEventdt[intraEventdt<0] *= -1
    return intraEventdt

def intra_event_dx(splitdf):
    """
    The geographic distance between each pair of the sequence of events given.
    
    Params:
    -------
    splitdf (list of DataFrames)

    Returns:
    --------
    intraEventdx (ndarray)
    """
    intraEventdx = []
    for s in splitdf:
        latlon = get_latlon(s)
        intraEventdx.append( pdist(latlon,euclid_dist) )
    intraEventdx = np.concatenate(intraEventdx)
    return intraEventdx

def extract_event_type(df, event_type):
    """
    Extract given event type from ACLED DataFrame. Some corrections to the data.
    
    Parameters
    ----------
    df : pd.DataFrame
    event_type : str
    
    Returns
    -------
    subdf : pd.DataFrame
    """
    ix=df['EVENT_TYPE'].apply(lambda x: event_type in x).values
    subdf=df.iloc[ix].copy()

    # Fix some typos. v5.0
    actorHeaders = ['ACTOR1','ALLY_ACTOR_1','ACTOR2','ALLY_ACTOR_2','ASSOC_ACTOR_1','ASSOC_ACTOR_2']
    for h in actorHeaders:
        if h in subdf.columns:
            subdf[h].fillna('',inplace=True)
            ix = (subdf[h]=='civililans (liberia)').values
            subdf.loc[ix,h] = 'civilians (liberia)'
            subdf[h] = subdf[h].apply(lambda x:'' if 'civilian' in x else x)
    return subdf

def digitize_actors(subdf):
    """Relabel actors by index instead of full name except for civilians who will be removed by
    replacing them with empty strings.
    
    First get the unique set of actors. Then relabel them.

    Parameters
    ----------
    subdf : pandas.DataFrame

    Returns
    -------
    subdf : pandas.DataFrame
        With new column 'actors' added.
    """
    uActors, uIx = np.unique(np.concatenate(( subdf['ACTOR1'], subdf['ALLY_ACTOR_1'],
                                              subdf['ACTOR2'], subdf['ALLY_ACTOR_2'] )),
                             return_index=True)
    uActors = uActors.tolist()
    print("Setting index 0 to this actors: %s"%uActors[0])

    subdf.loc[:,'ACTOR1'] = subdf['ACTOR1'].apply(lambda x: uActors.index(x))
    subdf.loc[:,'ALLY_ACTOR_1'] = subdf['ALLY_ACTOR_1'].apply(lambda x: uActors.index(x))
    subdf.loc[:,'ACTOR2'] = subdf['ACTOR2'].apply(lambda x: uActors.index(x))
    subdf.loc[:,'ALLY_ACTOR_2'] = subdf['ALLY_ACTOR_2'].apply(lambda x: uActors.index(x))

    # Define function for getting the indices of all actors in every event.
    filt_actor = lambda x:not (x==0 or np.isnan(x))
    def f(row):
        a = []
        if filt_actor(row['ACTOR1']):
            a.append(row['ACTOR1'])
        if filt_actor(row['ALLY_ACTOR_1']):
            a.append(row['ALLY_ACTOR_1'])
        if filt_actor(row['ACTOR2']):
            a.append(row['ACTOR2'])
        if filt_actor(row['ALLY_ACTOR_2']):
            a.append(row['ALLY_ACTOR_2'])
        if len(a)>0:
            a = sorted(a)
        return a

    # Must reset index in order to get new column rows to line up with old rows.
    subdf.reset_index(inplace=True)
    subdf['actors']=pd.Series([f(row) for i,row in subdf.iterrows()])
    return subdf

def plot_gmap(subdf,map_name='mymap.html'):
    """
    Draw on Google map.
    
    Params:
    -------
    subdf (pd.DataFrame)
    map_name (str='mymap.html')
    """
    latlon = get_latlon(subdf)

    gmap = gmplot.GoogleMapPlotter(latlon[:,0].mean(),latlon[:,1].mean(), 4)
    gmap.heatmap(latlon[:,0], latlon[:,1],opacity=.8)

    gmap.draw(map_name)

def gauss_smooth(x,width,windowlen):
    """
    Convolve with Gaussian for moving mean smoothing.
    
    Params:
    -------
    x (ndarray)
    width (ndarray)
    windowlen (int)

    Returns:
    --------
    xfilt (ndarray)
    """
    from scipy.signal import get_window,fftconvolve
    window = get_window(('gauss',width),windowlen)
    window /= window.sum()
    return fftconvolve(x,window,mode='same')
