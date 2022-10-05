# Author : Eddie Lee, edlee@csh.ac.at

from scipy.optimize import minimize
import numpy as np
from coniii.enumerate import fast_logsumexp



class NActivationIsing():
    def __init__(self, n, params=None, rng=None):
        """
        Parameters
        ----------
        int : n
            Number of spins, 1 + no. of neighbors.
        params : ndarray, None
        rng : np.random.RandomState, None
            Uses np.random if None.
        """

        self.n = n
        if not params is None:
            assert params.size==2 and params.ndim==1
            self.set_params(params)
        else:
            self.set_params(np.zeros(2))
        self.rng = rng if not rng is None else np.random
        
    def set_params(self, params):
        self.params = params
        
        self.logZ = fast_logsumexp(-self.calc_e(self.all_states(), self.params))[0]
        self.p = np.exp(-self.calc_e(self.all_states(), self.params) - self.logZ)
        
    def calc_e(self, s, params):
        return -self.params[0]*s[:,0] - self.params[1]*s[:,1:].sum(1)

    def calc_observables(self):
        return np.array([self.p[self.n:].sum(), self.p[self.n:].dot(np.arange(self.n))])

    def sample(self, size=1):
        """Sample from possible states using full probability distribution.

        Parameters
        ----------
        size : int, 1

        Returns
        -------
        ndarray
        """

        if size==1:
            return self.all_states()[self.rng.choice(2*self.n, p=self.p)][None,:]
        return self.all_states()[self.rng.choice(2*self.n, p=self.p, size=size)]
    
    def solve(self, constraints, original_guess=np.zeros(2), max_param_value=20):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(2)
        max_param_value : float, 20

        Returns
        -------
        dict
        """

        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            return np.linalg.norm(self.calc_observables() - constraints)
        return minimize(cost, original_guess,
                        bounds=[(-max_param_value, max_param_value)]*2)
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin. Second col is number of active neighbors.
        """

        return np.vstack(([0]*self.n + [1]*self.n, list(range(self.n)) + list(range(self.n)))).T
#end NActivationIsing
