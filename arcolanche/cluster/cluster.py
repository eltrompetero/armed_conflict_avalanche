# ====================================================================================== #
# Module for clustering routines used to generate avalanches.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from itertools import product
from functools import partial
import geopandas as gpd

from .voronoi import check_voronoi_tiles
from ..utils import *

def extend_poissd_coarse_grid(dx):
    """Redefine PoissonDiscSphere to consider an extended number of coarse grid
    neighbors to avoid boundary artifacts (that though uncommon) would manifest from
    not considering neighbors because of thin bounds on polygons.

    This increases the default number of coarse neighbors 9 used in PoissonDiscSphere
    to 15.

    Parameters
    ----------
    dx : int
    """
    
    for i in range(10):
        with open(f'voronoi_grids/{dx}/{str(i).zfill(2)}.p','rb') as f:
            poissd = pickle.load(f)['poissd']
            poissd = _extend_poissd_coarse_grid(poissd)
            pickle.dump({'poissd':poissd}, open(f'voronoi_grids/{dx}/{str(i).zfill(2)}.p.new','wb'))

def _extend_poissd_coarse_grid(poissd):
    """Main part of .extend_poissd_coarse_grid()

    Parameters
    ----------
    poissd : PoissonDiscSphere

    Returns
    -------
    PoissonDiscSphere
    """

    newpoissd = PoissonDiscSphere(poissd.r,
                                  width_bds=poissd.width,
                                  height_bds=poissd.height,
                                  coarse_grid=poissd.coarseGrid,
                                  k_coarse=15)
    newpoissd.samples = poissd.samples

    for i, s in enumerate(poissd.samples):
        newpoissd.samplesByGrid[poissd.assign_grid_point(s)].append(i)

    return newpoissd

def cluster_battles(iprint=True):
    """Generate conflict clusters across separation scales.

    Parameters
    ----------
    iprint : bool, True
    """

    from .workspace import load_battlesgdf

    assert os.path.isdir('cache/africa')

    def loop_wrapper(args, gridix=0):
        dx, dt = args
        
        battlesgdf = load_battlesgdf(dx, gridix)
        cellfile = f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp'
        polygons = gpd.read_file(cellfile)

        cellneighbors = {}
        for i, p in polygons.iterrows():
            # neighbors including self
            cellneighbors[i] = [int(i) for i in p['neighbors'].split(', ')] + [i]
        
        avalanches = cluster_avalanche(battlesgdf, dt, cellneighbors)
        return avalanches
    
    # Iterate thru combinations of separation scales and times
    # Separation scales are denoted by the inverse angle ratio used to to generate centers
    dxRange = [40, 80, 160, 320, 640, 1280]
    dtRange = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    with mp.Pool(mp.cpu_count()-1) as pool:
        for gridix in range(10):
            # first find all clusters of avalanches
            avalanches_ = pool.map(partial(loop_wrapper, gridix=gridix), product(dxRange, dtRange))
            
            # then collate results into dict indexed as (dx, dt)
            avalanches = {}
            for i, (dx, dt) in enumerate(product(dxRange, dtRange)):
                avalanches[(dx, dt)] = avalanches_[i]
            with open(f'cache/africa/battles_avalanche{str(gridix).zfill(2)}.p', 'wb') as f:
                pickle.dump({'avalanches':avalanches}, f)
                
            if iprint: print(f'Done with {gridix=}.')

def polygonize(iter_pairs=None):
    """Create polygons denoting boundaries of Voronoi grid.

    Parameters
    ----------
    iter_pairs : list of twoples, None
        Can be specified to direct polygonization for particular combinations of dx
        and grids {dx as int}, {gridix as int}. When None, goes through preset list
        of all combos from dx=80 up to dx=1280 (about 35km).
    """

    from numpy import pi
    from .voronoi import unwrap_lon, create_polygon

    def loop_wrapper(args):
        dx, gridix = args
        poissd = pickle.load(open(f'voronoi_grids/{dx}/{str(gridix).zfill(2)}.p', 'rb'))['poissd']

        # identify polygons that are within interesting boundaries
        lonlat = poissd.samples.copy()
        for i in range(len(lonlat)):
            lonlat[i] = unwrap_lon((lonlat[i,0]/pi*180 + 330)%360), lonlat[i,1]/pi*180
        if dx<=20:
            selectix = np.where((lonlat[:,0]>-37.2) & (lonlat[:,0]<70.5) &
                                (lonlat[:,1]>-54) & (lonlat[:,1]<57))[0]
        elif dx<=40:
            selectix = np.where((lonlat[:,0]>-22.2) & (lonlat[:,0]<55.5) &
                                (lonlat[:,1]>-39) & (lonlat[:,1]<42))[0]
        else:
            selectix = np.where((lonlat[:,0]>-19.7) & (lonlat[:,0]<53.5) &
                                (lonlat[:,1]>-37) & (lonlat[:,1]<40))[0]
        
        # create bounding polygons, the "Voronoi cells"
        polygons = []
        for i in selectix:
            try:
                polygons.append(create_polygon(poissd, i))
            except Exception:
                raise Exception(f"Problem with {i}.")
        polygons = gpd.GeoDataFrame({'index':list(range(len(polygons)))},
                                    geometry=polygons,
                                    crs='EPSG:4326')

        # identify all neighbors of each polygon
        neighbors = []
        sindex = polygons.sindex
        scaled_polygons = polygons['geometry'].scale(1.01,1.01)
        for i, p in polygons.iterrows():
            # scale polygons by a small factor to account for precision error in determining
            # neighboring polygons; especially important once dx becomes large, say 320
            # first lines look right but seem to involve some bug in detecting intersections
            #pseries = gpd.GeoSeries(p.geometry, crs=polygons.crs).scale(1.001, 1.001)
            #neighborix = sindex.query_bulk(pseries)[1].tolist()
            neighborix = np.where(polygons.intersects(scaled_polygons.iloc[i]))[0].tolist()

            # remove self
            try:
                neighborix.pop(neighborix.index(i))
            except ValueError:
                pass
            assert len(neighborix)

            # must save list as string for compatibility with Fiona pickling
            neighbors.append(str(sorted(neighborix))[1:-1])
        polygons['neighbors'] = neighbors

        # correct errors
        polygons, n_inconsis = check_voronoi_tiles(polygons, parallel=False)

        # save
        polygons.to_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
        
    if iter_pairs is None:
        # iterate over all preset combinations of dx and dt
        iter_pairs = product([40, 80, 160, 320, 640, 1280], range(10))

    with mp.Pool() as pool:
        pool.map(loop_wrapper, iter_pairs)

def cluster_avalanche(gdf, A, cellneighbors,
                      counter_mx=np.inf,
                      use_cpp=True,
                      run_checks=False):
    """Cluster events into avalanches by connecting all pixels that are neighbors in space
    that are active within the specified time interval A.
    
    Parameters
    ----------
    gdf : GeoDataFrame
    A : int
        Separation time scale
    cellneighbors : dict
        Key indicates which pixel and values indicate all neighboring pixels.
    counter_mx : int, np.inf
        If specified, only construct this number of avalanches.
    use_cpp : bool, True
    run_checks : bool, False
       
    Returns
    -------
    list of list of ints
        Each list indicates the indices of points that belong to one avalanche or another.
        These are simple ordered indices, not necessarily the "index" of the dataframe,
        i.e. use .iloc().
    """
    
    if use_cpp:
        from ..utils_ext import cluster_avalanche
        day = (gdf['event_date']-gdf['event_date'].min()).values / np.timedelta64(1,'D')
        day = day.astype(int)
        pixel = gdf['pixel'].values
        assert isinstance(cellneighbors, dict)

        avalanches = cluster_avalanche(day, pixel, A, cellneighbors, counter_mx)
    else:
        remaining = list(range(len(gdf)))  # unclustered event ix
        clustered = []  # clustered event ix
        avalanches = []  # groups of conflict avalanches of event ix

        counter = 0
        while remaining and counter<counter_mx:
            thisCluster = []
            toConsider = []  # events whose neighbors remain to be explored

            # initialize a cluster
            toConsider.append(remaining[0])

            while toConsider:
                # add this event to the cluster
                thisEvent = toConsider.pop(0)
                remaining.pop(remaining.index(thisEvent))
                thisCluster.append(thisEvent)
                clustered.append(thisEvent)
                thisPix = gdf['pixel'].iloc[thisEvent]

                # find all the neighbors of this point
                # first select all points within time dt
                selectix = np.zeros(len(gdf), dtype=bool)
                selectix[remaining] = np.abs(gdf['event_date'].iloc[remaining] -
                                             gdf['event_date'].iloc[thisEvent])<=np.timedelta64(A, 'D')

                # now select other events by cell adjacency
                for i in np.where(selectix)[0]:
                    if (gdf['pixel'].iloc[i] in cellneighbors[thisPix] and
                        not i in thisCluster and
                        not i in toConsider):
                        toConsider.append(i)

            avalanches.append(thisCluster)
            counter += 1
    
    if run_checks:
        # check that each event only appears once
        allix = np.concatenate(avalanches)
        assert allix.size==np.unique(allix).size, "Some events appear in more than once."
    
    return avalanches

def cluster_cells(cells, active):
    """Cluster voronoi cells by contiguity. This is basically the routine in
    .cluster_avalanche() except not accounting for time.
    
    Parameters
    ----------
    cells: GeoDataFrame
        Polygons that constitute Voronoi tessellation of map.
    active: ndarray
        Indices of cells that should be clustered by proximity.
       
    Returns
    -------
    list of list of ints
        Each list indicates the indices of points that belong to one avalanche or another.
        Since we are given a dataframe, the indices of the data rows will be given.
    """
    
    assert len(active)

    if isinstance(active, np.ndarray):
        remaining = active.tolist()  # unclustered event ix
    else:
        remaining = active[:]
    clustered = []  # clustered event ix
    avalanches = []  # groups of conflict avalanches of event ix

    counter = 0
    while remaining:
        thisCluster = []
        toConsider = []  # events whose neighbors remain to be explored

        # initialize a cluster
        toConsider.append(remaining[0])

        while toConsider:
            # add this event to the cluster
            thisEvent = toConsider.pop(0)
            remaining.pop(remaining.index(thisEvent))
            thisCluster.append(thisEvent)
            clustered.append(thisEvent)
            thisPix = cells.loc[thisEvent]

            # find all the neighbors of this point
            for i in thisPix.neighbors:
                if (i in remaining and
                    not i in thisCluster and
                    not i in toConsider):
                    toConsider.append(i)

        avalanches.append(thisCluster)
        counter += 1

    # check that each event only appears once
    allix = np.concatenate(avalanches)
    assert allix.size==np.unique(allix).size, "Some events appear in more than once."
    
    return avalanches

def revise_neighbors(dx, gridix, write=True):
    """Re-calculate neighbors and save to shapefile. Used for a one-time fix, but
    saved here for later reference.

    Parameters
    ----------
    dx : int
        Inverse solid angle.
    gridix : int
        Grid index.
    write : bool, True
        If True, write to shapely file.
    """

    assert os.path.isfile(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    polygons = gpd.read_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    
    # identify all neighbors of each polygon
    neighbors = []
    sindex = polygons.sindex
    for i, p in polygons.iterrows():
        # scale polygons by a small factor to account for precision error in determining
        # neighboring polygons; especially important once dx becomes large, say 320
        pseries = gpd.GeoSeries(p.geometry, crs=polygons.crs).scale(1.001, 1.001)
        neighborix = sindex.query_bulk(pseries)[1].tolist()

        # remove self
        try:
            neighborix.pop(neighborix.index(i))
        except ValueError:
            pass
        assert len(neighborix)

        # must save list as string for compatibility with Fiona pickling
        neighbors.append(str(sorted(neighborix))[1:-1])
    polygons['neighbors'] = neighbors

    polygons, n_inconsis = check_voronoi_tiles(polygons)
    if n_inconsis:
        print(f"There are {n_inconsis} asymmetric pairs that were corrected.")
    
    if write:
        polygons.to_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
    return polygons