# ====================================================================================== #
# For setting up a workspace for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *



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

