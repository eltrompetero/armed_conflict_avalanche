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

def check_voronoi_tiles(polygons, iprint=False, parallel=True):
    """Check Voronoi tiles to make sure that they are consistent.

    This will take any asymmetric pair of tiles (where one considers the other to be
    a neighbor but not vice versa) and make sure that both neighbors lists are
    consistent with one another.

    Parameters
    ----------
    polygons : geopandas.GeoDataFrame
    iprint : bool, False
    parallel : bool, False

    Returns
    -------
    geopandas.GeoDataFrame
    int
    """
    
    from shapely import wkt
    from shapely.errors import TopologicalError
    
    assert (polygons['index']==polygons.index).all()
    assert (np.diff(polygons['index'])==1).all()

    assert polygons.geometry.is_valid.all()
    if iprint: print("All geometries valid.")
    
    n_inconsis = 0
    for i, row in polygons.iterrows():
        for n in row['neighbors'].split(', '):
            n = int(n)
            if not str(i) in polygons.loc[n]['neighbors'].split(', '):
                new_neighbors = sorted(polygons.loc[n]['neighbors'].split(', ') + [str(i)])
                polygons.loc[n,'neighbors'] = ', '.join(new_neighbors)
                n_inconsis += 1
    if iprint: print("Done with correcting asymmetric neighbors.")

    # check overlap with all of africa
    # load africa
    africa = gpd.read_file(f'data/africa_countries/afr_g2014_2013_0.shp')
    assert africa.crs.name=='WGS 84'
    
    # drop island countries
    countries_to_drop = ['Cape Verde', 'Mauritius', 'Seychelles']
    keepix = np.ones(len(africa), dtype=bool)
    for c in countries_to_drop:
        keepix[africa['ADM0_NAME']==c] = False
    africa = africa.loc[keepix]

    # for each country check that intersection w/ voronoi area is very
    # close to total country, but these will not always be the same
    # b/c of precision error creating gaps or overlap btwn voronoi cells
    def loop_wrapper(args):
        i, country = args
        assert np.isclose(voronoi_cov.intersection(country.geometry).area, country.geometry.area,
                          rtol=1e-3), (i, country)
    
    if parallel:
        with Pool() as pool:
            # union voronoi cells
            voronoi_cov = polygons.iloc[0].geometry
            for i in range(1, len(polygons)):
                voronoi_cov = voronoi_cov.union(polygons.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            try:
                pool.map(loop_wrapper, africa.iterrows())
            except TopologicalError:
                polygons['geometry'] = polygons['geometry'].apply(lambda x:wkt.loads(wkt.dumps(x,
                                                                  rounding_precision=8)))
                # union voronoi cells
                voronoi_cov = polygons.iloc[0].geometry
                for i in range(1, len(polygons)):
                    voronoi_cov = voronoi_cov.union(polygons.iloc[i].geometry)
                voronoi_cov = gpd.GeoSeries(voronoi_cov)
                pool.map(loop_wrapper, africa.iterrows())
    else:
        try:
            # union voronoi cells
            voronoi_cov = polygons.iloc[0].geometry
            for i in range(1, len(polygons)):
                voronoi_cov = voronoi_cov.union(polygons.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            for args in africa.iterrows():
                loop_wrapper(args)
        except TopologicalError:
            polygons['geometry'] = polygons['geometry'].apply(lambda x:wkt.loads(wkt.dumps(x,
                                                                rounding_precision=8)))
            # union voronoi cells
            voronoi_cov = polygons.iloc[0].geometry
            for i in range(1, len(polygons)):
                voronoi_cov = voronoi_cov.union(polygons.iloc[i].geometry)
            voronoi_cov = gpd.GeoSeries(voronoi_cov)
            for args in africa.iterrows():
                loop_wrapper(args)
 
    if iprint: print("Done with checking overlap with Africa.")

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

