# ====================================================================================== #
# Causal avalanche network methods using transfer entropy.
# Author : Eddie Lee, Niraj Kushwaha
# ====================================================================================== #
import networkx as nx
from .transfer_entropy_func import iter_polygon_pair
from .self_loop_entropy_func import iter_valid_polygons

from .utils import *


def links(time_series, neighbor_info_dataframe,
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
    
    pair_poly_te = iter_polygon_pair(polygon_pair_gen(),
                                     number_of_shuffles, 
                                     time_series)
    return pair_poly_te


def self_links(time_series, number_of_shuffles=50):
    """Calculates self loop transfer entropy and identifies polygons with significant self loops assuming a 95% confidence interval.

    Parameters
    ----------
    time_series : pd.DataFrame
    number_of_shuffles : int, 50
    
    Returns
    -------
    pd.DataFrame
    pd.DataFrame
    list of tuples
        (cell index, cell index, TE)
        TE is nan when non-significant
    """  
    
    def valid_polygons_finder():
        valid_polygons = time_series.columns.astype(int).to_list()

        return valid_polygons

    valid_poly_te = iter_valid_polygons(valid_polygons_finder(),
                                        number_of_shuffles,
                                        time_series)

    return valid_poly_te


class CausalGraph():
    def __init__(self, self_poly_te, pair_poly_te, sig_threshold=95):
        """
        Parameters
        ----------
        self_poly_te : dict
            Keys are twoples. Values are TE and TE shuffles.
        pair_poly_te : dict
            Keys are twoples. Values are TE and TE shuffles.
        sig_threshold : float, 95
        """

        assert 0<=sig_threshold<=100 and isinstance(sig_threshold, int)
        self.self_poly_te = self_poly_te
        self.pair_poly_te = pair_poly_te
        self.sig_threshold = sig_threshold

        self.build_causal()
        
    def build_causal(self):
        """Build causal network using self.sig_threshold.

        This replaces self.G.
        """

        self.G = nx.DiGraph()
        for poly, (te, te_shuffle) in self.self_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.G.add_edge(poly, poly)

        for pair, (te, te_shuffle) in self.pair_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.G.add_edge(pair[0], pair[1])

        self.uG = self.G.to_undirected()

        
# end CausalGraph    
    #    # process output into convenient packaging
    #    clean_pair_poly_te = []
    #    filtered_neighbors = {}
    #    for key, val in pair_poly_te.items():
    #        # check if polygon already in dict
    #        if not key[0] in filtered_neighbors.keys():
    #            filtered_neighbors[key[0]] = []

    #        # add sig neighbors
    #        if (val[0]>val[1]).mean()>.95:
    #            filtered_neighbors[key[0]].append(key[1])
    #            clean_pair_poly_te.append((key[0], key[1], val[0]))
    #        else:
    #            clean_pair_poly_te.append((key[0], key[1], np.nan))

