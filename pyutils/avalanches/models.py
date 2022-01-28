# ====================================================================================== #
# Models for simulating avalanches on Voronoi cell activity.
# This was written for binary coarse-graining of activity (is site active or inactive) for
# a local maxent formulation.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from ..utils import *
from coniii.utils import bin_states
from coniii.enumerate import fast_logsumexp



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
            if [len(i)-2 for i in self.fields.values()]==[len(i) for i in self.neighbors.values()]:
                warn("Assuming that spin bias was specified.")
                self.step = self.step_bias
            else:
                assert [len(i)-1 for i in self.fields.values()]==[len(i) for i in self.neighbors.values()]
        elif self.__class__ is LocalMaxent21:
            pass

    def _step(self, replace_active_sites=True, inactive_value=-1):
        """Advance simulation by one time step.
        
        Iterate thru all sites and consider the probability that they should be activated
        at the next time step, which is determined by conditioning on the state of the
        neighbors and calculating the relatively probability that the pixel is active.
        
        Parameters
        ----------
        replace_active_sites : bool, True
        inactive_value : int, -1
        
        Returns
        -------
        list of ints
            New list of active sites. This is a reference to the data member.
        """
        
        newActiveSites = []
        
        for pix, h in self.fields.items():
            # reset previous state to all inactive
            prevState = np.zeros(h.size, dtype=int) + inactive_value

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
 
    def step(self, replace_active_sites=True, inactive_value=0):
        """Advance simulation by one time step.
        
        Iterate thru all sites and consider the probability that they should be activated
        at the next time step, which is determined by conditioning on the state of the
        neighbors and calculating the relatively probability that the pixel is active.
        
        Parameters
        ----------
        replace_active_sites : bool, True
        inactive_value : int, -1
        
        Returns
        -------
        list of ints
            New list of active sites. This is a reference to the data member.
        """
        
        newActiveSites = []
        
        for pix, h in self.fields.items():
            # reset previous state to all inactive
            prevState = np.zeros(h.size, dtype=int) + inactive_value

            # was this pixel active in previous time step?
            if pix in self.activeSites:
                prevState[0] = 1
                
            # fetch config of neighboring cells
            for i, nix in enumerate(self.neighbors[pix]):
                if nix in self.activeSites:
                    prevState[i+1] = 1

            #pa = h.dot(prevState) - sum([fast_logsumexp([log3, h_])[0] for h_ in h])
            pa = h.dot(prevState) - np.log(3 + np.exp(h)).sum()
            pa = np.exp(pa)
            print(pa)
            if self.rng.rand() < pa:
                newActiveSites.append(pix)
        
        if replace_active_sites:
            self.activeSites = set(newActiveSites)
        return newActiveSites
    
    def sample(self, n_iters, initial_state=None, inactive_value=-1):
        """Use jit to generate avalanche.

        Parameters
        ----------
        n_iters : int
        initial_state : set, None

        Returns
        -------
        numba.List of sets
        """

        from numba.typed import List, Dict
        from numba.core import types
        
        pix = List(self.fields.keys())
        h = List(self.fields.values())
        neighbors = Dict.empty(key_type=types.int64, value_type=types.int64[:])
        for k, v in self.neighbors.items():
            neighbors[k] = np.array(v, dtype=np.int64)
        
        @njit
        def step(activeSites, pix, h, neighbors):
            newActiveSites = List()
            
            for pix_, h_ in zip(pix, h):
                # reset previous state to all inactive
                prevState = np.zeros(h_.size) + inactive_value

                # was this pixel active in previous time step?
                if pix_ in activeSites:
                    prevState[0] = 1
                    
                # fetch config of neighboring cells
                for i, nix in enumerate(neighbors[pix_]):
                    if nix in activeSites:
                        prevState[i+1] = 1

                pa = (np.tanh(prevState.dot(h_)) + 1) / 2
                    
                if np.random.rand() < pa:
                    newActiveSites.append(pix_)
                
            # always include -1 to avoid empty list
            if not len(newActiveSites):
                return set((-1,))
            return set(newActiveSites)
        
        if initial_state is None:
            activeSites = [set((-1,))]
        else:
            activeSites = [initial_state]

        for i in range(n_iters):
            activeSites.append(step(activeSites[-1], pix, h, neighbors))
        
        return [set() if i=={-1} else i for i in activeSites]

    def step_bias(self, replace_active_sites=True):
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
            prevState = np.zeros(h.size-1, dtype=int) - 1

            # was this pixel active in previous time step?
            if pix in self.activeSites:
                prevState[0] = 1
                
            # fetch config of neighboring cells
            for i, nix in enumerate(self.neighbors[pix]):
                if nix in self.activeSites:
                    prevState[i+1] = 1

            pa = (np.tanh(h[0] + prevState.dot(h[1:])) + 1) / 2
                
            if self.rng.rand() < pa:
                newActiveSites.append(pix)
        
        if replace_active_sites:
            self.activeSites = set(newActiveSites)
        return newActiveSites
    
    @staticmethod
    def pair_corr(traj, i, j, dt=0, sym=True):
        """Extract pair correlation between sites i and j from avalanche trajectory
        potentially with a time delay.
        
        Parameters
        ----------
        traj : list of ints
            Each element lists all sites that were active in a particular time step. All
            unnamed sites are inactive.
        i : int
            Pixel in future.
        j : int
            Pixel in past.
        dt : int, 0
            Size of lag.
        sym : bool, True
        
        Returns
        -------
        float
            Return correlation in {-1,1} basis unless sym is False.
        """
        
        assert isinstance(dt, int) and len(traj)>dt>=0
    
        if sym:
            c = 0
            for t in range(dt, len(traj)):
                iIsIn = i in traj[t]
                jIsIn = j in traj[t-dt]
                if not (iIsIn^jIsIn):
                    c += 1
                else:
                    c -= 1
        else:
            c = 0
            for t in range(dt, len(traj)):
                iIsIn = i in traj[t]
                jIsIn = j in traj[t-dt]
                if iIsIn and jIsIn:
                    c += 1
 
        c /= len(traj)-dt
        return c
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
