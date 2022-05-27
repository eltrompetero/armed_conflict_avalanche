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
    dict
        dict with keys as directed edge and
        values as tuple where first element is
        the transfer entropy and the second
        element is a list of shuffled transfer entropies.
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
    dict
        dict with keys as self loop tiles and
        values as tuple where first element is
        the self transfer entropy and the second
        element is a list of shuffled self transfer entropies.
    """  
    
    def valid_polygons_finder():
        valid_polygons = time_series.columns.astype(int).to_list()

        return valid_polygons

    valid_poly_te = iter_valid_polygons(valid_polygons_finder(),
                                        number_of_shuffles,
                                        time_series)

    return valid_poly_te


class CausalGraph(nx.DiGraph):
    def setup(self, self_poly_te, pair_poly_te, sig_threshold=95):
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

        """

        for poly, (te, te_shuffle) in self.self_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.add_edge(poly, poly)

        for pair, (te, te_shuffle) in self.pair_poly_te.items():
            if (te>te_shuffle).mean() >= (self.sig_threshold/100):
                self.add_edge(pair[0], pair[1])

        self.uG = self.to_undirected()


    def self_loop_list(self):
        """Outputs a list of all nodes which have a self loop.

        Returns
        -------
        list
            A list of all nodes which have a self loop.

        """
        self_loop_node_list = []
        for i in self.edges:
            if(i[0] == i[1]):
                self_loop_node_list.append(i[0])

        return self_loop_node_list


    def edges_no_self(self):
        """Outputs a list of tuples where each tuple contains node index which has a
         causal link between them.

        Returns
        -------
        list
            A list of tuples where each tuple contain two nodes which have a link
            between them. 
        """
        return [i for i in self.edges() if i[0] != i[1]]


    def causal_neighbors(self):
        """Outputs a dict where keys are node index and values are list of
         successive neighbors.
        """
        neighbor_dict = {}
        for node in self.nodes:
            neighbor_list_temp = []
            for neighbor in self.successors(node):
                if(node != neighbor):
                    neighbor_list_temp.append(neighbor)
                neighbor_dict[node] = neighbor_list_temp

        return neighbor_dict


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

