# =============================================================================================== #
# Tools for analyzing AWD CSV data from the Zammit-Mignon PNAS article.
# Author: Eddie Lee
# Python v3.6
# =============================================================================================== #
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
DATADR = os.path.expanduser('~')+'/Dropbox/Research/armed_conflict/data/'



#def euclid_dist(xxx_todo_changeme, xxx_todo_changeme1):
#    (lat, lng) = xxx_todo_changeme
#    (lat0, lng0) = xxx_todo_changeme1
#    deglen = 110.25
#    x = lat - lat0
#    y = (lng - lng0)*np.cos(lat0)
#    return deglen*np.sqrt(x*x + y*y)
#
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

def split_by_date(t,threshold=14):
    """Iterate through list of times and split by where the thresholds are.
    
    Parameters
    ----------
    t : list of ndarrays
        Each subarray must be sorted.
    threshold : float
        Max positive difference allowed.

    Returns
    -------
    t : list of ndarrays
    """
    result=[]
    for i,t_ in enumerate(t):
        dt=np.diff(t_)
        assert (dt>=0).all()
        result.append( np.split( t_,np.where(np.diff(t_)>threshold)[0]+1 ) )
    return result

#def split_by_dist(df,threshold=2,fast=False):
#    """Split data frame into list of pieces that are within a threshold distance of each other."""
#    dmat = np.zeros((len(df),len(df)))
#    latlon = np.vstack((df['LATITUDE'],df['LONGITUDE'])).T
#    if fast:
#        for i,j in combinations(list(range(len(dmat))),2):
#            dmat[i,j] = euclid_dist(latlon[i],latlon[j])
#    else:
#        for i,j in combinations(list(range(len(dmat))),2):
#            dmat[i,j] = great_circle(latlon[i],latlon[j]).meters/1000
#    dmat += dmat.T
#    
#    split = []
#    clustix = cluster_dist(dmat,threshold)
#    for ix in clustix:
#        split.append(df.iloc[ix])
#    return split
#
#def pipeline_civ_violence(df):
#    """Extract violence against civilians events."""
#    subdf = df.iloc[(df['EVENT_TYPE']=='Violence against civilians').values]
#
#    split = []
#    for a in unique(subdf['ACTOR1']):
#        split.append(subdf.iloc[(subdf['ACTOR1']==a).values])
#        split[-1].sort_values('EVENT_DATE',inplace=True)
#
#    split_ = split
#
#    # Split by date.
#    split = [split_by_date(s) if (len(s)>1) else [s] for s in split]
#    split = list(chain.from_iterable(split))
#
#    # Split by distance
#    split = [split_by_dist(s) if (len(s)>1) else [s] for s in split]
#    split = list(chain.from_iterable(split))
#    
#    return split
#
#def merge(sets):
#    """Merge a list of sets such that any two sets with any intersection are merged."""
#    merged = 1
#    while merged:
#        merged = 0
#        results = []
#        while sets:
#            common, rest = sets[0], sets[1:]
#            sets = []
#            for x in rest:
#                if x.isdisjoint(common):
#                    sets.append(x)
#                else:
#                    merged = 1
#                    common |= x
#            results.append(common)
#        sets = results
#    return sets
#
## Split by participants.
#def split_by_actor(subdf):
#    uactors = []
#    for a in subdf['actors']:
#        if not a in uactors:
#            uactors.append(a)
#    
#    # Combine all sets that intersect.
#    uactors = [set(a) for a in uactors]
#    if len(uactors)==1:
#        return [subdf]
#    uactors = merge(uactors)
#    
#    split = [[] for i in range(len(uactors))]
#    for ai,a in enumerate(subdf['actors']):
#        for si,s in enumerate(uactors):
#            if set(a)<=s:
#                split[si].append(subdf.iloc[ai])
#                break
#    split = [pd.concat(s,axis=1,ignore_index=False).transpose() for s in split]
#    return split
#

def cluster_by_latlon(latlon,mxdist):
    """
    Use BallTree clustering to cluster events by their latitude and longitude. The algorithm
    iterates through every point given and finds the neighbors within mxdist. All unique sets of
    events are found and returned.

    Parameters
    ----------
    latlon : ndarray
    mxdist : float
        Max distance in km used to decide which events belong together.

    Returns
    -------
    clusterIx : list of ndarray
        Each ndarray lists the indices of the events that belong in that cluster.
    """
    latlon = latlon*np.pi/180  # convert to radians
    tree = BallTree(latlon,leaf_size=20,metric='haversine')
    
    clusteredEventsIx = []
    for ll in latlon:
        # d is distance between this point and neighbors. In order to keep things efficient, I only
        # return a maximum of 100 neighbors.
        d,ix = tree.query([ll],k=min([len(latlon),100]))
        ix = ix.ravel()[d.ravel()<(mxdist/6.378e3)]

        # If the number of neighbors exceeds the cutoff, then we need to expand our ball to include
        # more neighbors.
        cutoff = len(d[0])
        while len(ix)==cutoff and cutoff!=len(latlon):
            cutoff = min([cutoff*2,len(latlon)])
            d,ix = tree.query([ll],k=cutoff)
            ix = ix.ravel()[d.ravel()<(mxdist/6.378e3)]
        
        clusteredEventsIx.append(np.sort(ix.tolist()))
    
    # Only keep unique clusters.
    uClusteredEventsIx=[list(x) for x in set(tuple(x) for x in clusteredEventsIx)]
    return uClusteredEventsIx

#def split_by_dist_fast(subdf,mxdist=40,method='manual'):
#    latlon = get_latlon(subdf)
#    
#    if method=='manual':
#        uClusteredEventsIx = cluster_by_latlon(latlon,mxdist)
#    elif method=='cluster':
#        ms = MeanShift(bandwidth=mxdist/6.378e3)
#        ms.fit(latlon*np.pi/180)
#        uClusteredEventsIx = []
#        for i in np.unique(ms.labels_):
#            uClusteredEventsIx.append( np.where(ms.labels_==i)[0] )
#    else:
#        raise Exception("Invalid method.")
#    
#    splitdf = []
#    for c in uClusteredEventsIx:
#        splitdf.append(subdf.iloc[c])
#    return splitdf
#
#def center(X,ddof=0):
#    m = X.mean()
#    width = X.std(ddof=ddof)
#    
#    return (X-m)/width,m,width
#
#def square_fit_mean_std(m,s,return_all=False):
#    """
#    Fit a square root curve for the standard deviation given the mean with the
#    assumption that these should be related to each other by a scaling
#    constant.
#
#    Params:
#    -------
#    m (array-like)
#        Means.
#    s (array-like)
#        Standard deviations.
#
#    Returns:
#    --------
#    Constant factor.
#    """
#    def cost(a):
#        return ((a*np.sqrt(m)-s)**2).sum()
#    if return_all:
#        return minimize(cost,1)
#    else:
#        return minimize(cost,1)['x']
#
#def square_val_mean_std(sqfit,m):
#    return np.sqrt(m)*sqfit
#
#def square_fit(x,y,weights=None):
#    """
#    Fit a function of the form a*sqrt(x)+b with max likelihood and option to
#    provide weights on data points. Usually the weight is proportional to the
#    number of data points.
#
#    Params:
#    -------
#    x,y (array-like)
#    weights (array-like=None)
#
#    Returns:
#    --------
#    sqfit
#        (a,b)
#    """
#    if weights is None:
#        weights = np.ones_like(y)
#
#    def cost(params):
#        a,b = params
#        return (( (a*np.sqrt(x)+b-y)**2 )*weights).sum()
#    return minimize(cost,[0,0])['x']
#
#def square_val(sqfit,x):
#    return sqfit[0]*np.sqrt(x) + sqfit[1]
#
#def intra_event_dt(splitdf,pairwise=False):
#    """
#    The number of days between sequential events of the sequence of events given or all pairs of events.
#    
#    Params:
#    -------
#    splitdf (list of DataFrames)
#    pairwise (bool=False)
#        If True, find the absolute value of the time differences between all pairs of events.
#
#    Returns:
#    --------
#    intraEventdt (ndarray)
#    """
#    intraEventdt = []
#    if pairwise:
#        d = lambda u,v: (v[3]-u[3])/np.timedelta64(1,'D')
#        for s in splitdf:
#            intraEventdt.append( pdist(s,d) )
#    else:
#        # Assuming the sequence has been ordered.
#        for s in splitdf:
#            intraEventdt.append( np.diff(s['EVENT_DATE'])/np.timedelta64(1,'D') )
#
#    intraEventdt = np.concatenate(intraEventdt).astype(int)
#    intraEventdt[intraEventdt<0] *= -1
#    return intraEventdt
#
#def intra_event_dx(splitdf):
#    """
#    The geographic distance between each pair of the sequence of events given.
#    
#    Params:
#    -------
#    splitdf (list of DataFrames)
#
#    Returns:
#    --------
#    intraEventdx (ndarray)
#    """
#    intraEventdx = []
#    for s in splitdf:
#        latlon = get_latlon(s)
#        intraEventdx.append( pdist(latlon,euclid_dist) )
#    intraEventdx = np.concatenate(intraEventdx)
#    return intraEventdx
#
#def extract_event_type(df,event_type):
#    """
#    Extract given event type from ACLED DataFrame.
#    
#    Parameters
#    ----------
#    df : pd.DataFrame
#    event_type : str
#    
#    Returns
#    -------
#    subdf (pd.DataFrame)
#    """
#    ix=df['EVENT_TYPE'].apply(lambda x: event_type in x).values
#    subdf=df.iloc[ix].copy()
#
#    # Don't consider civilians as actors (do this by replacing them with empty strings). 
#    # Fix some typos. v5.0
#    actorHeaders = ['ACTOR1','ALLY_ACTOR_1','ACTOR2','ALLY_ACTOR_2']
#    for h in actorHeaders:
#        subdf[h].fillna('',inplace=True)
#        ix = (subdf[h]=='Civililans (Liberia)').values
#        subdf[h].iloc[ix] = 'Civilians (Liberia)'
#        subdf[h].values[:] = subdf[h].apply(lambda x:'' if 'Civilian' in x else x)
#
#    # Relabel actors by index.
#    uActors,uIx = np.unique(np.concatenate(( subdf['ACTOR1'],subdf['ALLY_ACTOR_1'],
#                                             subdf['ACTOR2'],subdf['ALLY_ACTOR_2'] )),
#                            return_index=True)
#    uActors = uActors.tolist()
#
#
#    subdf['ACTOR1'].values[:] = subdf['ACTOR1'].apply(lambda x: uActors.index(x))
#    subdf['ALLY_ACTOR_1'].values[:] = subdf['ALLY_ACTOR_1'].apply(lambda x: uActors.index(x))
#    subdf['ACTOR2'].values[:] = subdf['ACTOR2'].apply(lambda x: uActors.index(x))
#    subdf['ALLY_ACTOR_2'].values[:] = subdf['ALLY_ACTOR_2'].apply(lambda x: uActors.index(x))
#
#    # Concatenate actors into a single list.
#    filt_actor = lambda x:not (x==0 or np.isnan(x))
#    def f(row):
#        a = []
#        if filt_actor(row['ACTOR1']):
#            a.append(row['ACTOR1'])
#        if filt_actor(row['ALLY_ACTOR_1']):
#            a.append(row['ALLY_ACTOR_1'])
#        if filt_actor(row['ACTOR2']):
#            a.append(row['ACTOR2'])
#        if filt_actor(row['ALLY_ACTOR_2']):
#            a.append(row['ALLY_ACTOR_2'])
#        if len(a)>0:
#            a = np.sort(a).tolist()
#        return a
#    subdf['actors']=pd.Series([f(subdf.iloc[i]) for i in range(len(subdf))])
#    return subdf
#
#def plot_gmap(subdf,map_name='mymap.html'):
#    """
#    Draw on Google map.
#    
#    Params:
#    -------
#    subdf (pd.DataFrame)
#    map_name (str='mymap.html')
#    """
#    latlon = get_latlon(subdf)
#
#    gmap = gmplot.GoogleMapPlotter(latlon[:,0].mean(),latlon[:,1].mean(), 4)
#    gmap.heatmap(latlon[:,0], latlon[:,1],opacity=.8)
#
#    gmap.draw(map_name)
#
#def gauss_smooth(x,width,windowlen):
#    """
#    Convolve with Gaussian for moving mean smoothing.
#    
#    Params:
#    -------
#    x (ndarray)
#    width (ndarray)
#    windowlen (int)
#
#    Returns:
#    --------
#    xfilt (ndarray)
#    """
#    from scipy.signal import get_window,fftconvolve
#    window = get_window(('gauss',width),windowlen)
#    window /= window.sum()
#    return fftconvolve(x,window,mode='same')
#
