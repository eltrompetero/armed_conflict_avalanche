# ====================================================================================== #
# For setting up a workspace for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from itertools import product



def grid_of(var, gridix=0, use_cache=True):
    """Extract "scaling variables" from conflict avalanches. These include R (reports), F
    (fatalities), T (duration), N (sites), A (road area). Results are cached because they
    are slow to load.
    
    Parameters
    ----------
    var : str
    gridix : int, 0
    use_cache : bool, True
    
    Returns
    -------
    dict
    """

    if os.path.isfile(f'cache/africa/grid{var}.p'):
        return pickle.load(open(f'cache/africa/grid{var}.p','rb'))[var]

    avalanches = pickle.load(open(f'cache/africa/battles_avalanche{str(gridix).zfill(2)}.p','rb'))['avalanches']
    if var=='R':
        R = {}
        for k, a in avalanches.items():
            R[k] = np.array([len(i) for i in a])
        pickle.dump({'R':R}, open('cache/africa/gridR.p','wb'))
        return R
    else:
        if var=='F':
            F = {}
            for k, a in avalanches.items():
                battlesgdf = load_battlesgdf(k[0], gridix)
                F[k] = np.array([battlesgdf.fatalities.iloc[ix].sum() for ix in a])
            pickle.dump({'F':F}, open('cache/africa/gridF.p','wb'))
            return F
        elif var=='N':
            N = {}
            for k, a in avalanches.items():
                battlesgdf = load_battlesgdf(k[0], gridix)
                N[k] = np.array([np.unique(battlesgdf.pixel.iloc[ix]).size for ix in a])
            pickle.dump({'N':N}, open('cache/africa/gridN.p','wb'))
            return N
        elif var=='T':
            T = {}
            for k, a in avalanches.items():
                battlesgdf = load_battlesgdf(k[0], gridix)
                T[k] = np.array([(battlesgdf.event_date.iloc[ix].max()-battlesgdf.event_date.iloc[ix].min()).days
                                 for ix in a])
            pickle.dump({'T':T}, open('cache/africa/gridT.p','wb'))
            return T
        elif var=='A':
            A = {}
            raster = rasterio.open('africa_roads/combined_.01.tif', )
            for k, a in avalanches.items():
                battlesgdf = load_battlesgdf(k[0], gridix)
                cellfile = f'voronoi_grids/{k[0]}/borders{str(gridix).zfill(2)}.shp'
                # take the mean over the area first to normalize by pixel area
                rdensity = np.array([i['mean'] if i['mean'] else 0 for i in 
                                     zonal_stats(cellfile, 'africa_roads/combined_.01.tif',
                                                 band=1,
                                                 stats=['mean'])])
                # total road "area" in all pixels entered
                A[k] = np.array([rdensity[np.unique(battlesgdf.iloc[ix].pixel)].sum()
                                 for ix in a])
            pickle.dump({'A':A}, open('cache/africa/gridA.p','wb'))
            return A
    raise NotImplementedError

def load_battlesgdf(dx, ix=0):
    """Load specified GDF. This includes mapping to appropriate Voronoi cells in the
    'pixel' column.

    Parameters
    ----------
    dx : int
    ix : int, 0

    Returns
    -------
    GeoDataFrame
    """
    
    return pickle.load(open(f'voronoi_grids/{dx}/battlesgdf{str(ix).zfill(2)}.p', 'rb'))['battlesgdf']

def setup_battlesgdf(iprint=False):
    """Modify battles DataFrame for use with GeoDataFrame. This will add pixel column to
    the dataframe.
    """
    
    from data_sets.acled import ACLED2020

    def loop_wrapper(args):
        dx, gridix = args
        cellfile = f'voronoi_grids/{dx}/borders{str(gridix).zfill(2)}.shp'
        polygons = gpd.read_file(cellfile)

        # data
        battlesdf = ACLED2020.battles_df()
        battlesgdf = gpd.GeoDataFrame(battlesdf,
                                      geometry=gpd.points_from_xy(battlesdf.longitude,
                                                                  battlesdf.latitude))

        # assign each event to a cell/pixel
        pixel = np.zeros(len(battlesgdf), dtype=int)

        for i, p in polygons.iterrows():
            ix = battlesgdf.geometry.within(p.geometry)
            pixel[ix] = i    
        battlesgdf['pixel'] = pixel

        with open(f'voronoi_grids/{dx}/battlesgdf{str(gridix).zfill(2)}.p', 'wb') as f:
            pickle.dump({'battlesgdf':battlesgdf}, f)
        if iprint: print(f'Done with battlesgdf{str(gridix).zfill(2)}.p')

    #pairs = product([40,80,160,320,640,1280], range(10))
    pairs = product([80], range(10))
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.map(loop_wrapper, pairs)

def default_dr(event_type='battle'):
    """Return default directory for event type.

    Parameters
    ----------
    event_type : str, 'battle'

    Returns
    -------
    str
    """
    
    if event_type=='battle':
        return f'{DEFAULTDR}/geosplits/africa/battle/full_data_set/bin_agg'
    elif event_type=='civ_violence':
        return f'{DEFAULTDR}/geosplits/africa/civ_violence/full_data_set/bin_agg'
    elif event_type=='riots':
        return f'{DEFAULTDR}/geosplits/africa/riots/full_data_set/bin_agg'
    else: raise Exception("Unrecognized event type.")

def load_default_pickles(event_type='battle', gridno=0):
    """For shortening the preamble on most Jupyter notebooks.
    
    Parameters
    ----------
    event_type : str, 'battle'
        Choose of 'battle', 'civ_violence', 'riots'.
    gridno : int, 0
        Index of cached Voronoi grid used.

    Returns
    -------
    pd.DataFrame
        subdf
    dict of lists
        gridOfSplits
    """

    # voronoi binary aggregation
    region = 'africa'
    prefix = 'voronoi_noactor_'
    method = 'voronoi'
    folder = f'geosplits/{region}/{event_type}/full_data_set/bin_agg'

    # Load data
    subdf = pickle.load(open(f'{DEFAULTDR}/{folder}/{prefix}df.p','rb'))['subdf']
    L = 9  # separation length index
    T = 11  # separation time index

    fname = f'{DEFAULTDR}/{folder}/{prefix}grid{str(gridno).zfill(2)}.p'
    gridOfSplits = pickle.load(open(fname,'rb'))['gridOfSplits']
    
    return subdf, gridOfSplits

