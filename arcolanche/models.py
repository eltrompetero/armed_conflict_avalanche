# Author : Eddie Lee, edlee@csh.ac.at

from scipy.optimize import minimize
import numpy as np
from coniii.enumerate import fast_logsumexp
from multiprocess import Pool


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
        """Re-defines self.logZ and self.p using given parameters.
        
        Parameters
        ----------
        params : ndarray
        """
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
    
    def solve(self, constraints,
              original_guess=np.zeros(2),
              max_param_value=20,
              J_cost=(0,50)):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(2)
        max_param_value : float, 20
        J_cost : twople, (0,50)
            (mean, std) of coupling cost

        Returns
        -------
        dict
        """

        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            return (np.linalg.norm(self.calc_observables() - constraints) +
                    abs(J_cost[0]-new_params[1])/J_cost[1])
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



def solve_simultaneous_activation(polygons, conf_df):
    """Simultaneous activation model.
    """

    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / tmx

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """

        pi = px.loc[i]
        pin = 0
        for t, g in g_by_t.groups.items():
            n_active = 0
            g_x = conf_df['x'].loc[g].values
            for j in polygons['active_neighbors'].loc[i]:
                if j in g_x:
                    n_active += 1
            pin += n_active
        pin /= tmx
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NActivationIsing(n)
        constraints = np.array([pi, pin])
        solver.solve(constraints, max_param_value=10, J_cost=(0, 10))
        return solver.params

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(0);
    loop_wrapper(polygons.index[0])
    #with Pool() as pool:
    #    polygons['params'] = list(pool.map(loop_wrapper, polygons.index))

    # read out parmeters into separate cols
    polygons['h'] = polygons['params'].apply(lambda i:i[0])
    polygons['J'] = polygons['params'].apply(lambda i:i[1])

def solve_delayed_activation(polygons, conf_df):
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / tmx

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """

        pi = px.loc[i]
        pin = 0

        # for each time point, count number of neighbors at t-1, assume that the rest are 0
        for t, g in g_by_t:
            n_active = 0
            if t-1 in g_by_t.groups.keys():
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        n_active += 1
            pin += n_active
        pin /= tmx
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NActivationIsing(n)
        constraints = np.array([pi, pin])
        solver.solve(constraints, max_param_value=10, J_cost=(0, 10))
        return solver.params

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(0);
    with Pool() as pool:
        polygons['params'] = list(pool.map(loop_wrapper, polygons.index))

    # read out parmeters into separate cols
    polygons['h'] = polygons['params'].apply(lambda i:i[0])
    polygons['J'] = polygons['params'].apply(lambda i:i[1])

