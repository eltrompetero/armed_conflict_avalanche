# Causal avalanche network using transfer entropy.
from .utils import *


def te_causal_network(time_series, neighbor_info_dataframe,
                      number_of_shuffles=50):
    """Calculates transfer entropy and identifies significant links between Voronoi
    neighbors assuming a 95% confidence interval.

    Parameters
    ----------
    time_series : pd.DataFrame
    neighbor_info_dataframe : pd.DataFrame
    number_of_shuffles : int, 50
    
    Returns
    -------
    pd.DataFrame
    pd.DataFrame
    list of tuples
        (cell index, cell index, TE)
        TE is nan when non-significant
    """

    # calculate transfer entropy between pairs of tiles
    def polygon_pair_gen():
        """Pairs of legitimate neighboring polygons."""
        for i, row in neighbor_info_dataframe.iterrows():
            for n in row['neighbors']:
                # only consider pairs of polygons that appear in the time series
                if row['index'] in time_series.columns and n in time_series.columns:
                    yield (row['index'], n)
    
    pair_poly_te = transfer_entropy_func.iter_polygon_pair(polygon_pair_gen(),
                                                           number_of_shuffles, 
                                                           time_series)
    # process output into convenient packaging
    clean_pair_poly_te = []
    filtered_neighbors = {}
    for key, val in pair_poly_te.items():
        # check if polygon already in dict
        if not key[0] in filtered_neighbors.keys():
            filtered_neighbors[key[0]] = []

        # add sig neighbors
        if (val[0]>val[1]).mean()>.95:
            filtered_neighbors[key[0]].append(key[1])
            clean_pair_poly_te.append((key[0], key[1], val[0]))
        else:
            clean_pair_poly_te.append((key[0], key[1], np.nan))

    return pair_poly_te, filtered_neighbors, clean_pair_poly_te

