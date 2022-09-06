# ====================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from scipy.spatial.distance import pdist
import pandas as pd
import geopandas as gpd
from datetime import datetime
from itertools import combinations, chain
from scipy.spatial.distance import squareform
from misc.plot import colorcycle
import scipy.stats as stats
from warnings import warn
from misc.globe import SphereCoordinate, PoissonDiscSphere, haversine, jithaversine
from numba import jit, njit
import dill as pickle
import multiprocess as mp
from statsmodels.distributions import ECDF
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterstats import zonal_stats, gen_zonal_stats
import rasterio
from threadpoolctl import threadpool_limits
from multiprocess import Pool

DEFAULTDR = os.path.expanduser('~')+'/Dropbox/Research/armed_conflict2/py'
DATADR = os.path.expanduser('~')+'/Dropbox/Research/armed_conflict2/data'




def wrap_lon(x):
    """Wrap longitude from [0,360] to interval [-180,180] while mapping 0 to 0.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    ndarray
    """
    
    assert (x>=0).all()

    x = x.copy()
    while (x>180).any():
        x[x>180] = x[x>180] - 360
    return x

def pixels_in_africa(polygons, countries_to_exclude=[]):
    """Select Voronoi cells that are in Africa.

    Parameters
    ----------
    polygons : geopandas.GeoDataFrame
    countries_to_exclue : list of str, []
    
    Returns
    -------
    geopandas.GeoDataFrame
    """
    
    # load countries and grab those in Africa
    countrygdf = gpd.read_file('../data/countries/ne_10m_admin_0_countries.shp')
    countrygdf = countrygdf.iloc[(countrygdf.CONTINENT=='Africa').values]
    
    # exclude countries should not be counted
    for c in countries_to_exclude:
        assert c in countrygdf.SOVEREIGNT.values, f"Country to exclude {c} not found."
        countrygdf.drop(countrygdf.index[np.where(countrygdf.SOVEREIGNT==c)[0][0]], inplace=True)

    # select polygons that intersect with specified countries
    selectix = np.zeros(len(polygons), dtype=bool)
    for i, p in polygons.iterrows():
        selectix[i] = countrygdf.intersects(p.geometry).any()

    return polygons.iloc[selectix]

def grid_split2cluster_ix(gridsplit, dfix):
    """Convert grid split (which indicates the cluster to which each event from the
    DataFrame belongs) into a column Series that can be added to the DataFrame.

    This facilitates use of native pandas functions for event analysis.
    
    Parameters
    ----------
    gridsplit : list of lists
        Each internal list is a conflict avalanche listing the DataFrame index of the
        events that belong to it. Note that this is not necessarily the order in which
        events are listed in the DataFrame.
    dfix : pd.Index
        Index of the DataFrame.

    Returns
    -------
    pd.Series
    """
    
    # indicates conflict avalanche to which each conflict event belongs
    clusterix = np.zeros(dfix.size, dtype=int)
    
    for avalancheix, split in enumerate(gridsplit):
        for splitix in split:
            # find the conflict event in df that corresponds to index in gridsplit assign
            # to it the conflict avalanche index "avalancheix"
            clusterix[np.where(dfix==splitix)[0][0]] = avalancheix

    return pd.Series(clusterix, index=dfix)

def track_max_pair_dist(lonlat,
                        as_delta=True,
                        use_pdist=False):
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
    """Check the scaling relation betwenn scaling variables X (along x-axis) and Y (along y-axis).
    Distribution of X has exponent alpha.
    Distribution of Y has exponent upsilon.
    Ratio of fractal dimension Y to X is df.
    For example: When Y is fatalities and X is time,
    the relation checked for this case would be
        $\\alpha - 1 = (\\upsilon-1) d_{\\rm f}$

    Parameters
    ----------
    alphaBds : tuple
        Lower and upper bounds for alpha.
    upsBds : tuple
        Lower and upper bounds for upsilon.
    dfBds : ndarray
        Lower and upper bounds for the ratio of fractal dimensions.

    Returns
    -------
    bool
        True if the scaling relation is violated.
    """

    if not (hasattr(alphaBds,'__len__') and hasattr(upsBds,'__len__')):
        return False
    if ( (((alphaBds[0]-1)>(dfBds[0]*(upsBds[0]-1))) &
          ((alphaBds[0]-1)<(dfBds[1]*(upsBds[1]-1)))) or
         (((alphaBds[1]-1)>(dfBds[0]*(upsBds[0]-1))) &
          ((alphaBds[1]-1)<(dfBds[1]*(upsBds[1]-1)))) or
         (((alphaBds[0]-1)<(dfBds[0]*(upsBds[0]-1))) &
          ((alphaBds[1]-1)>(dfBds[0]*(upsBds[0]-1)))) or
         (((alphaBds[0]-1)<(dfBds[1]*(upsBds[1]-1))) &
          ((alphaBds[1]-1)>(dfBds[1]*(upsBds[1]-1)))) ):
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

    Returns
    -------
    ndarray
        Pixel diameters for each space threshold given.
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

def merge(sets):
    """Merge a list of sets such that any two sets with any intersection are merged.

    Parameters
    ----------
    list of sets

    Returns
    -------
    list of sets
    """

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

