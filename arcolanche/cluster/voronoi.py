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
    assert len(neighborsix)>=3, "Provided point has less than three neighbors."

    center = SphereCoordinate(center[0], center[1]+pi/2)
    neighbors = [SphereCoordinate(s[0], s[1]+pi/2) for s in poissd.samples[neighborsix]]
    
    try:
        precision = 1e-7
        cell = VoronoiCell(center, rng=np.random.RandomState(0), precision=precision)
        triIx = cell.initialize_with_tri(neighbors)
    except AssertionError:
        # try reducing precision
        precision = 5e-8
        cell = VoronoiCell(center, rng=np.random.RandomState(0), precision=precision)
        triIx = cell.initialize_with_tri(neighbors)

    for i, ix in enumerate(triIx):
        neighbors.pop(ix-i)
    
    # iterate thru all neighbors and try to add them to convex hull, most won't be added
    for n in neighbors:
        cell.add_cut(GreatCircle.bisector(n, center))
    
    poly = Polygon([(unwrap_lon((v.phi/pi*180+330)%360), (v.theta-pi/2)/pi*180) for v in cell.vertices])
    return poly

def check_voronoi_tiles(polygons):
    """Check Voronoi tiles to make sure that they are consistent.

    This will take any asymmetric pair of tiles (where one considers the other to be
    a neighbor but not vice versa) and make sure that both neighbors lists are
    consistent with one another.

    Parameters
    ----------
    polygons : geopandas.GeoDataFrame
    """
    
    assert (polygons['index']==polygons.index).all()
    assert (np.diff(polygons['index'])==1).all()
    
    n_inconsis = 0
    for i, row in polygons.iterrows():
        for n in row['neighbors'].split(', '):
            n = int(n)
            if not str(i) in polygons.loc[n]['neighbors'].split(', '):
                new_neighbors = sorted(polygons.loc[n]['neighbors'].split(', ') + [str(i)])
                polygons.loc[n,'neighbors'] = ', '.join(new_neighbors)
                n_inconsis += 1
    return polygons, n_inconsis

def check_poisson_disc(poissd, min_dx):
    """Check PoissonDiscSphere grid.

    Parameters
    ----------
    poissd : PoissonDiscSphere
    min_dx : float
    """
    
    # min distance surpasses min radius
    for xy in poissd.samples:
        neighbors, dist = poissd.neighbors(xy, return_dist=True)
        zeroix = dist==0
        assert zeroix.sum()==1
        assert dist[~zeroix].min()>=min_dx, (min_dx, dist[~zeroix].min())

