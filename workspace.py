# ====================================================================================== #
# For setting up a workspace for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from itertools import product



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
    pairs = product([40], range(10))
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

