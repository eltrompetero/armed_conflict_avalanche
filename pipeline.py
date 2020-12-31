# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from itertools import product
from functools import partial
import geopandas as gpd




def extend_poissd_coarse_grid(dx):
    """Redefine PoissonDiscSphere to consider an extended number of coarse grid neighbors
    to avoid boundary artifacts (that though uncommon) would manifest from not considering
    neighbors because of thin bounds on polygons.

    This increases the default number of coarse neighbors 9 used in PoissonDiscSphere to
    15.

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
    """Generate conflict clusters across separation scales."""

    from .workspace import load_battlesgdf

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

def polygonize_voronoi(iter_pairs=None):
    """Create polygons denoting boundaries of Voronoi grid.

    Parameters
    ----------
    iter_pairs : list of twoples, None
        Can be specified to direct polygonization for particular combinations of dx and grids.
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
        if dx<=40:
            selectix = np.where((lonlat[:,0]>-20.2) & (lonlat[:,0]<53.5) &
                                (lonlat[:,1]>-39) & (lonlat[:,1]<42))[0]
        else:
            selectix = np.where((lonlat[:,0]>-18.7) & (lonlat[:,0]<52) &
                                (lonlat[:,1]>-36) & (lonlat[:,1]<40))[0]

        polygons = [create_polygon(poissd, i) for i in selectix]
        polygons = gpd.GeoDataFrame({'index':list(range(len(polygons)))},
                                    geometry=polygons,
                                    crs='EPSG:4087')

        # identify all neighbors of each polygon
        neighbors = []
        sindex = polygons.sindex
        for i, p in polygons.iterrows():
            # scale polygons by a small factor to account for precision error in determining
            # neighboring polygons; especially important once dx becomes large, say 320
            pseries = gpd.GeoSeries(p.geometry, crs=polygons.crs).scale(1.01)
            neighborix = sindex.query_bulk(pseries)[1].tolist()

            # remove self
            neighborix.pop(neighborix.index(i))
            assert len(neighborix)

            # must save list as string for compatibility with Fiona pickling
            neighbors.append(str(sorted(neighborix))[1:-1])
        polygons['neighbors'] = neighbors

        # save
        polygons.to_file(f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp')
        
    if iter_pairs is None:
        iter_pairs = product([80, 160, 320, 640, 1280], range(10))

    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.map(loop_wrapper, iter_pairs)

def rate_dynamics(dxdt=((160,16), (160,32), (160,64), (160,128), (160,256))):
    """Extract exponents from rate dynamical profiles.

    Automating analysis in "2019-11-18 rate dynamics.ipynb".

    Parameters
    ----------
    dxdt : tuple, ((160,16), (160,32), (160,64), (160,128), (160,256))

    Returns
    -------
    ndarray
        Dynamical exponent d_S/z.
    ndarray
        Dynamical exponent d_F/z.
    ndarray
        Dynamical exponent 1/z.
    """

    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist
    from .rate_dynamics import extract_from_df, interp_clusters

    dsz = np.zeros(len(dxdt))
    dfz = np.zeros(len(dxdt))
    invz = np.zeros(len(dxdt))
    subdf, gridOfSplits = load_default_pickles()

    for counter,dxdt_ in enumerate(dxdt):
        clusters = extract_from_df(subdf, gridOfSplits[dxdt_])

        # logarithmically spaced bins
        xinterp = np.linspace(0, 1, 100)
        bins = 2**np.arange(2, 14)
        npoints = np.arange(3, 20)[:bins.size-1]
        data, traj, clusterix = interp_clusters(clusters, xinterp, bins, npoints)


        offset = int(np.log2(dxdt_[1]))-2  # number of small bins to skip (skipping all avalanches with
                                           # duration <a days)

        # use variance in log space
        # these aren't weighted by number of data points because that returns something
        # similar and is more complicated
        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['S'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dsz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['F'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dfz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['L'][i+offset][1]/bins[i+offset]**a)
                
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        invz[counter] = minimize(cost, .5)['x']
        
        print("dxdt = ",dxdt_)
        print('ds/z = %1.2f'%dsz[counter])
        print('df/z = %1.2f'%dfz[counter])
        print('1/z = %1.2f'%invz[counter])
        print()
    
    return dsz, dfz, invz

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
