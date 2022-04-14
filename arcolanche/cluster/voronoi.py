# ====================================================================================== #
# Creating and handling voronoi tiling.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from ..utils import *
from itertools import product
from numpy import pi
from misc.globe import VoronoiCell, SphereCoordinate, GreatCircle
from shapely.geometry import Point, Polygon



def transform(phitheta):
    """From angles to lon, lat coordinates accounting for the longitudinal shift necessary
    to get to Africa.
    
    Parameters
    ----------
    phitheta : ndarray
    
    Returns
    -------
    ndarray
        Lonlat.
    """
    
    if phitheta.ndim==1:
        newcoord = phitheta / pi * 180
        newcoord[0] += 330
        return newcoord
        
    newcoord = phitheta / pi * 180
    newcoord[:,0] += 330
    return newcoord

def unwrap_lon(x):
    """Transform longitude from (0,360) to (-180,180).
    
    Parameters
    ----------
    x : ndarray or float
    
    Returns
    -------
    ndarray or float
    """
    
    if isinstance(x, np.ndarray):
        x = x.copy()
        ix = x>180
        x[ix] -= 180
        x[ix] *= -1
        return x
    
    if x>180:
        return -(360-x)
    return x

def create_polygon(poissd, centerix):
    """Construct polygon about specified point in PoissonDiscSphere.

    Parameters
    ----------
    poissd : PoissonDiscSphere
    centerix : int
        Construct polygon about specified point in PoissonDiscSphere.

    Returns
    -------
    shapely.geometry.Polygon
    """
    
    center = poissd.samples[centerix]

    neighborsix = poissd.neighbors(center)
    neighborsix.pop(neighborsix.index(centerix))

    center = SphereCoordinate(center[0], center[1]+pi/2)
    neighbors = [SphereCoordinate(s[0], s[1]+pi/2) for s in poissd.samples[neighborsix]]

    cell = VoronoiCell(center, rng=np.random.RandomState(0))
    triIx = cell.initialize_with_tri(neighbors)
    for i, ix in enumerate(triIx):
        neighbors.pop(ix-i)

    for n in neighbors:
        cell.add_cut(GreatCircle.bisector(n, center))
    
    poly = Polygon([(unwrap_lon((v.phi/pi*180+330)%360), (v.theta-pi/2)/pi*180) for v in cell.vertices])
    return poly
