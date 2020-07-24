# ====================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from sklearn.cluster import MeanShift
from scipy.spatial.distance import pdist
import pandas as pd
import os
from datetime import datetime
from geopy.distance import vincenty, great_circle
from itertools import combinations, chain
from scipy.spatial.distance import squareform
from misc.plot import colorcycle
from misc.utils import ortho_plane, max_dist_pair2D
import scipy.stats as stats
from scipy.optimize import minimize
from warnings import warn
from misc.globe import SphereCoordinate, PoissonDiscSphere, haversine, jithaversine
from numba import jit, njit
import pickle
DATADR = os.path.expanduser('~')+'/Dropbox/Research/armed_conflict/data/'



def track_max_pair_dist(lonlat, as_delta=True, use_pdist=False):
    """Keep track of spatial extent of cluster. By default, returns delta growth.
    
    Parameters
    ----------
    lonlat : ndarray
        Time-ordered list of geographic coordinates.
    as_delta : bool, True
    use_pdist : bool, False
        Slow way of computing every pairwise distance.

    Returns
    -------
    ndarray
        Distance per lonlat.
    """

    phitheta = lonlat/180*np.pi
    phitheta[:,1] += np.pi/2
    
    maxDistPair = [phitheta[0],phitheta[1]]
    maxdist = np.zeros(len(phitheta))  # always start at 0
    maxdist[1] = haversine(phitheta[0], phitheta[1])
    
    if not use_pdist:
        for i in range(2, maxdist.size):
            newdist = _max_pair_dist(lonlat[:i+1])
            if newdist>maxdist[i-1]:
                maxdist[i] = newdist
            else:
                maxdist[i] = maxdist[i-1]
    else:
        # this algorithm is slow (but guaranteed to be correct)
        for i in range(2, maxdist.size):
            phitheta_ = np.unique(phitheta[:i+1], axis=0)
            if len(phitheta_)>1:
                newdist = pdist(phitheta_, jithaversine).max() 
            else:
                newdist = 0
            if newdist>maxdist[i-1]:
                maxdist[i] = newdist
            else:
                maxdist[i] = maxdist[i-1]
    
    if as_delta:
        return np.insert(np.diff(maxdist), 0, 0)
    return maxdist

def _max_pair_dist(lonlat):
    from misc.globe import max_geodist_pair

    lonlat = np.unique(lonlat, axis=0)
    if len(lonlat)==1:
        return 0.

    # convert coordinates into a 3D vector
    phitheta = lonlat / 180 * np.pi
    phitheta[:,1] += np.pi/2
    maxdistix = max_geodist_pair(phitheta)

    return jithaversine(phitheta[maxdistix[0]], phitheta[maxdistix[1]])

def check_relation(alphaBds, upsBds, dfBds):
    """
    Checks the basic relation between the time scales exponent alpha with another scaling
    variable like fatalities. The relation checked for this case would be
        $\\alpha - 1 = (\\upsilon-1) d_{\\rm f}$

    Parameters
    ----------
    alphaBds : tuple
        Bound for alpha, exponent for P(T) ~ T^-alpha.
    upsBds : tuple
        Bound for variable to scale with T, like upsilon for fatalities.
    dfBds : ndarray
        Bounds for scaling of second variable with T (fractal dimension). Each sequential col
        represents lower then upper bounds respectively.

    Returns
    -------
    bool
        Whether or not the scaling relation is violated.
    """

    if not (hasattr(alphaBds,'__len__') and hasattr(upsBds,'__len__')):
        return False
    if ( (((alphaBds[0]-1)>(dfBds[:,0]*(upsBds[0]-1))) & ((alphaBds[0]-1)<(dfBds[:,1]*(upsBds[1]-1)))).any() or
         (((alphaBds[1]-1)>(dfBds[:,0]*(upsBds[0]-1))) & ((alphaBds[1]-1)<(dfBds[:,1]*(upsBds[1]-1)))).any() or
         (((alphaBds[0]-1)<(dfBds[:,0]*(upsBds[0]-1))) & ((alphaBds[1]-1)>(dfBds[:,0]*(upsBds[0]-1)))).any() or
         (((alphaBds[0]-1)<(dfBds[:,1]*(upsBds[1]-1))) & ((alphaBds[1]-1)>(dfBds[:,1]*(upsBds[1]-1)))).any() ):
        return True
    return False

def exponent_bounds(alphaBds, upsBds, dfBds):
    """Checks the basic relation between the time scales exponent alpha with another scaling
    variable like fatalities. The relation checked for this case would be
        $\\alpha - 1 = (\\upsilon-1) d_{\\rm f}$

    Show max range possible for third exponent given other two.

    Parameters
    ----------
    alphaBds : tuple
        Bound for alpha, exponent for P(T) ~ T^-alpha.
    upsBds : tuple
        Bound for variable to scale with T, like upsilon for fatalities.
    dfBds : tuple

    Returns
    -------
    tuples
    """

    dfRange = (alphaBds[0]-1)/(upsBds[1]-1), (alphaBds[1]-1)/(upsBds[0]-1)
    alphaRange = dfBds[0]*(upsBds[0]-1)+1, dfBds[1]*(upsBds[1]-1)+1
    upsRange = (alphaBds[0]-1)/dfBds[1]+1, (alphaBds[1]-1)/dfBds[0]+1
    return alphaRange, upsRange, dfRange

def percentile_bds(X, perc, as_delta=False):
    """Percentile range.
    
    Just a wrapper around np.percentile to make it easier."""

    if hasattr(X,'__len__') and (not np.isnan(X).any()):
        if as_delta:
            if type(as_delta) is bool:
                return (np.percentile(X,50)-np.percentile(X, perc[0]),
                        np.percentile(X,perc[1])-np.percentile(X,50))
            return (as_delta-np.percentile(X, perc[0]),
                    np.percentile(X,perc[1])-as_delta)
        return np.percentile(X, perc[0]), np.percentile(X, perc[1])
    return None

def sigma_bds(X, factor=1.):
    if hasattr(X,'__len__'):
        mu = np.mean(X)
        sigma = np.std(X)
        return (mu-sigma*factor, mu+sigma*factor)
    return None

def transform_bds_to_offset(x,bds):
    """To make it easier to use matplotlib's errorbar plotting function, take a vector of
    data and error bounds and convert it to an output that can be passed directly to the
    yerr kwarg.
    """
    
    if type(x) is float or type(x) is int:
        assert bds.size==2
        return np.array([[x-bds[0]],bds[1]-x])
    
    return np.vstack((x-bds[0],bds[1]-x))

def coarse_grain_voronoi_tess(dx, fileno):
    """Successive coarse-graining from lowest layer to each upper layers. To keep the
    coarse-graining consistent, coarse-graining happens only between adjacent levels and
    then the entire coarse-graining operation is traced from the bottom to the top layer.

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

    assert (np.diff(dx)<0).all(), "Grid must get coarser."
    
    # mappings between adjacent layers
    nextLayerPixel = _coarse_grain_voronoi_tess(dx, fileno)
    
    # Iterate through layers and track to back bottom-most layer
    bottomLayerMapping = [nextLayerPixel[0]]
    for el in range(1, len(dx)-1):
        bottomLayerMapping.append(nextLayerPixel[el][bottomLayerMapping[el-1]])
    return bottomLayerMapping

def _coarse_grain_voronoi_tess(dx, fileno):
    """Coarse-graining by adjacent layers.

    Parameters
    ----------
    dx : list
        Spacing of adjacent layers with which to coarse grain.
    fileno : int
        Only file with this name will be taken from all layers specified in dx.
        
    Returns
    -------
    nextLayerPixel : list
        Each list maps dx[i] to dx[i+1] by pixel index in dx[i+1], i.e. each entry names
        the coarse-grained pixel to which it belongs such that the length of this array is
        as long as the number of pixels in the fine-grained layer.
    """

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

def voronoi_pix_diameter(spaceThreshold, n_samp=10):
    """Get an estimate of the average distance between the centers of Voronoi tiles by
    loading grid 00.p for each resolution specified.

    Parameters
    ----------
    spaceThreshold : list
        The space threshold parameter known as "dx".
        Ranges from 1280 to 5 in factors of 2.
    n_samp : int, 10
        Number of random pairs to take average of.
    """

    pixDiameter = np.zeros(len(spaceThreshold))
    for i,dx in enumerate(spaceThreshold):
        # there is a factor of 2 in how spacing is determined between points in Voronoi
        # algo because any neighboring "tiles" within 2r are considered to belong to the
        # same cluster
        pixDiameter[i] = _sample_lattice_spacing(dx, n_samp) * 2
    return pixDiameter

def _sample_lattice_spacing(dx, sample_size):
    """
    Parameters
    ----------
    dx : float
        Separation length.
    sample_size : int
        No of random points to use to estimate radius (by taking mean).
        
    Returns
    -------
    dist : float
        Distance from one point to its closest neighbor (km).
    """

    poissd = pickle.load(open('voronoi_grids/%d/00.p'%dx,'rb'))['poissd']
    
    if sample_size>len(poissd.samples):
        sample_size=len(poissd.samples)
        
    randix = np.random.choice(np.arange(len(poissd.samples)), size=sample_size, replace=True)

    d = np.zeros(sample_size)
    for i,ix in enumerate(randix):
        d[i] = poissd.closest_neighbor_dist(poissd.samples[ix])

    return d.mean() * 6370

def loglog_fit_err_bars(x, y, fit_params, show_plot=False):
    """Calculate posterior probability of exponent parameter.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    fit_params : twople
    posterior : bool, False
    show_plot : bool, False

    Returns
    -------
    twople
        95% confidence intervals on exponent parameter assuming fixed offset.
    """

    from numpy import log
    from misc.stats import loglog_fit

    # posterior probability estimation of error bars
    fit_params=loglog_fit(x, y)

    resx=log(y) - np.polyval(fit_params, log(x))
    resy=(log(y) - fit_params[1])/fit_params[0] - log(x)
    varerr=np.concatenate((resx, resy)).var(ddof=1)

    def f(s, t=fit_params[1], x=x, y=y):
        """Function for calculating log likelihood."""
        return -1/2/varerr * ( ((log(y) - s*log(x) - t)**2 + ((log(y)-t)/s - log(x))**2).mean() )
    f=np.vectorize(f)

    # find bounding interval corresponding to a drop of exp(10) in probability
    dx=1e-2  # amount to increase bounds by per iteration
    bds=[fit_params[0], fit_params[0]]
    peak=f(fit_params[0])

    while (peak-f(bds[0]))<10:
        bds[0]-=dx
        
    while (peak-f(bds[1]))<10:
        bds[1]+=dx
    
    # construct discrete approximation to probability distribution
    x=np.linspace(*bds, 10_000)
    y=f(x)
    y-=y.max()
    p=np.exp(y)
    p/=p.sum()

    if show_plot:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.plot(x, p)
        ax.vlines(fit_params[0], 0, p.max())
        ax.set(xlabel=r'$x$')
        ax.legend((r'$p(x)$', r'$s^*$'))

    # sample for confidence intervals
    r=np.random.choice(x, p=p, size=1_000_000)

    if show_plot:
        return (np.percentile(r,2.5), np.percentile(r,97.5)), (fig,ax)
    return np.percentile(r,2.5), np.percentile(r,97.5)

def split_by_nan(v):
    """Split a vector into non-nan segments that are separated by nans.
    
    Parameters
    ----------
    v : ndarray
    
    Returns
    -------
    v_separated : list
        Each element is a vector of a contiguous sequence of non-nan numbers.
    ix : list
        Each element is the index of the elements that correspond to the given elements in
        the first variable.
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

def split_by_index(ix,X):
    """Split given dataframe at specified indices.
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

def get_latlon(df):
    return np.vstack((df['latitude'], df['longitude'])).T

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
            intraEventdt.append( np.diff(s['event_date'])/np.timedelta64(1,'D') )

    intraEventdt = np.concatenate(intraEventdt).astype(int)
    intraEventdt[intraEventdt<0] *= -1
    return intraEventdt

def intra_event_dx(splitdf):
    """The geographic distance between each pair of the sequence of events given.
    
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

    from scipy.signal import get_window, fftconvolve
    window = get_window(('gauss',width), windowlen)
    window /= window.sum()
    return fftconvolve(x, window, mode='same')
