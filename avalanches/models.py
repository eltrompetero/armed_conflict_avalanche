# ====================================================================================== #
# Models for simulating avalanches on Voronoi cell activity.
# This was written for binary coarse-graining of activity (is site active or inactive) for
# a local maxent formulation.
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
class LocalMaxent11():
    """Run avalanche simulation using local maxent probability distribution fixing
    correlations between center site x_t and the past of self and neighbors x_{t-1} and
    y_{t-1}, also referred to as the (1,1) model.
    
    Starting configuration is all sites being at 0 since self.activeSites is initialized
    as empty.
    """
    def __init__(self, fields, neighbors, rng=None):
        """
        Parameters
        ----------
        fields : dict
            Indexed by pixel, where each value is the field that the spin of row index
            feels from neighbors.
        neighbors : GeoDataFrame or dict
            Specifies the indices of all neighbors for each pixel in stateprobs.
        rng : np.RandomState, None
        """
        
        # read in args
        self.fields = fields
        self.neighbors = neighbors
        self.rng = rng or np.random
        
        # check args
        if isinstance(self.neighbors, pd.Series):
            self.neighbors = self.neighbors.to_dict()
        assert isinstance(self.neighbors, dict), "neighbors must be dict or translatable to one"
        
        # set up model
        self.activeSites = []

        if self.__class__ is LocalMaxent11:
            assert [len(i)-1 for i in self.fields.values()]==[len(i) for i in self.neighbors.values()]
        elif self.__class__ is LocalMaxent21:
            pass
        
    def step(self, replace_active_sites=True):
        """Advance simulation by one time step.
        
        Iterate thru all sites and consider the probability that they should be activated
        at the next time step, which is determined by conditioning on the state of the
        neighbors and calculating the relatively probability that the pixel is active.
        
        Parameters
        ----------
        replace_active_sites : bool, True
        
        Returns
        -------
        list of ints
            New list of active sites. This is a reference to the data member.
        """
        
        newActiveSites = []
        
        for pix, h in self.fields.items():
            # reset previous state to all -1 (inactive)
            prevState = np.zeros(h.size, dtype=int) - 1

            # was this pixel active in previous time step?
            if pix in self.activeSites:
                prevState[0] = 1
                
            # fetch config of neighboring cells
            for i, nix in enumerate(self.neighbors[pix]):
                if nix in self.activeSites:
                    prevState[i+1] = 1

            pa = (np.tanh(prevState.dot(h)) + 1) / 2
                
            if self.rng.rand() < pa:
                newActiveSites.append(pix)
        
        if replace_active_sites:
            self.activeSites = set(newActiveSites)
        return newActiveSites
#end LocalMaxent11


class LocalMaxent21(LocalMaxent11):
    """Run avalanche simulation using local maxent probability distribution fixing
    correlations between center site x_t and the past of self and pairs of neighbors
    x_{t-1} and y_{t-1}, also referred to as the (2,1) model.
    
    Starting configuration is all sites being at 0 since self.activeSites is initialized
    as empty.
    """
    def step(self, replace_active_sites=True):
        """Advance simulation by one time step.
        
        Iterate thru all sites and consider the probability that they should be activated
        at the next time step, which is determined by conditioning on the state of the
        neighbors and calculating the relatively probability that the pixel is active.
        
        Parameters
        ----------
        replace_active_sites : bool, True
        
        Returns
        -------
        list of ints
            New list of active sites. This is a reference to the data member.
        """
        
        newActiveSites = []
        
        for pix, params in self.fields.items():
            h, J = params
            n = h.size

            # reset previous state to all -1
            prevState = np.zeros(n) - 1

            # was this pixel active in previous time step?
            if pix in self.activeSites:
                prevState[0] = 1
                
            # fetch config of neighboring cells
            for i, nix in enumerate(self.neighbors[pix]):
                if nix in self.activeSites:
                    prevState[i+1] = 1

            pa = (np.tanh(prevState.dot(h + self.pairwise_prod(prevState[1:].dot(J)))) + 1) / 2
                
            if self.rng.rand() < pa:
                newActiveSites.append(pix)
        
        if replace_active_sites:
            self.activeSites = set(newActiveSites)
        return newActiveSites
    
    @staticmethod
    def pairwise_prod(x):
        """Pairwise products of all spins in x.

        This could be sped up using jit and a for loop.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """
        
        ix = np.vstack(list(combinations(x.size, 2)))
        return x[ix[:,0]] * x[ix[:,1]]
#end LocalMaxent21
