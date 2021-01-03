# ====================================================================================== #
# Models for simulating avalanches on Voronoi cell activity.
# This was written for binary coarse-graining of activity for a local maxent formulation.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from ..utils import *
from coniii.utils import bin_states



# ========= #
# Functions #
# ========= #



# ======= #
# Classes #
# ======= #
class LocalMaxent1():
    """Run avalanche simulation using local maxent probability distribution fixing
    correlations between center site x_t and the past of self and neighbors x_{t-1} and
    y_{t-1}.
    
    Starting configuration is all sites being at 0 since self.activeSites is initialized
    as empty.
    """
    def __init__(self, stateprobs, neighbors, rng=None):
        """
        Parameters
        ----------
        stateprobs : dict of tuples
            First element is an integer for system size, the total no. of
            "spins" accounted for.
            Second element is an ndarray for all state probabilities.
        neighbors : GeoDataFrame or dict
            Specifies the indices of all neighbors for each pixel in stateprobs.
        rng : np.RandomState, None
        """
        
        # read in args
        self.stateprobs = stateprobs
        self.neighbors = neighbors
        self.rng = rng or np.random
        
        # check args
        assert all([len(i)==2 for i in stateprobs.values()]), "something wrong with stateprobs"
        assert all([isinstance(i[0], int) for i in stateprobs.values()]), "system size must be int"
        if isinstance(self.neighbors, pd.Series):
            self.neighbors = self.neighbors.to_dict()
        assert isinstance(self.neighbors, dict), "neighbors must be dict or translatable to one"
        assert all([(len(i)+2)==self.stateprobs[k][0]
                    for k, i in self.neighbors.items()]), "neighbors don't match up"
        
        # set up model
        self.activeSites = []
        self.states = dict([(n, bin_states(n)) for n in range(2, 12)])
        
    def step(self, replace_active_sites=True):
        """Advance simulation by one time step.
        
        Iterate thru all sites and consider the probability that they should be activated
        at the next time step.
        
        Parameters
        ----------
        replace_active_sites : bool, True
        
        Returns
        -------
        list of ints
            New list of active sites. This is a reference to the data member.
        """
        
        newActiveSites = []
        
        for pix, (n, pall) in self.stateprobs.items():
            if not pix in self.activeSites:
                # fetch config of neighboring cells
                neighState = np.zeros((1, n), dtype=int)
                for i, nix in enumerate(self.neighbors[pix]):
                    if nix in self.activeSites:
                        neighState[0,i+2] = 1
                
                # normalize active prob conditional on neighbor state
                pa = 0.
                pa += pall[(neighState==self.states[n]).all(1)]
                neighState[0,0] = 1
                pa += pall[(neighState==self.states[n]).all(1)]
                pa = pall[(neighState==self.states[n]).all(1)] / pa
                
            else:
                # fetch config of neighboring cells
                neighState = np.zeros((1, n), dtype=int)
                neighState[0,1] = 1
                for i, nix in enumerate(self.neighbors[pix]):
                    if nix in self.activeSites:
                        neighState[0,i+2] = 1
                
                # normalize active prob conditional on neighbor state
                pa = 0.
                pa += pall[(neighState==self.states[n]).all(1)]
                neighState[0,0] = 1
                pa += pall[(neighState==self.states[n]).all(1)]
                pa = pall[(neighState==self.states[n]).all(1)] / pa
                
            if self.rng.rand() < pa:
                newActiveSites.append(pix)
        
        if replace_active_sites:
            self.activeSites = newActiveSites
        return newActiveSites
#end LocalMaxent1
