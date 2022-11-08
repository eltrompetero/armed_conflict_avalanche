# Author : Eddie Lee, edlee@csh.ac.at

from scipy.optimize import minimize
from coniii.enumerate import fast_logsumexp

from .construct import discretize_conflict_events
from .utils import *


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
        for t,g in g_by_t.groups.items():
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

def solve_delayed_activation(polygons, conf_df, K_cost_inverse_weight=10):
    """Solve for the parameters of a delayed activation model given the data.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'J'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
    """
    
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    # corr w/ self in the future
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:self_corr(i-tmn, tmx-tmn))

    def loop_wrapper(i):
        """For a given site, fit its self correlation with its pair correlations no.
        of active neighbors.
        """

        pi = px.loc[i]
        pin = 0

        # for each time point, count number of neighbors at t-1 only if i was active
        # at t, assume that the rest are 0
        for t, g in g_by_t:
            n_active = 0
            if i in g['x'].values and t-1 in g_by_t.groups.keys():
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        n_active += 1
            pin += n_active
        pin /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        solver = NActivationIsing(n)
        constraints = np.array([pi, pin])
        solver.solve(constraints, max_param_value=20, K_cost=(0, K_cost_inverse_weight))
        return solver.params

    g_by_t = conf_df.groupby('t')  # all conflicts grouped by time bin
    g_by_t.get_group(conf_df['t'].iloc[0]);  # cache groups
    with Pool() as pool:
        polygons['params'] = list(pool.map(loop_wrapper, polygons.index))

    # read out parmeters into separate cols
    polygons['h'] = polygons['params'].apply(lambda i:i[0])
    polygons['J'] = polygons['params'].apply(lambda i:i[1])

def self_corr(t, norm):
    """Probability of an active state following an active state.

    Parameters
    ----------
    t : np.ndarray
        Unique times at which activity was observed. Min possible is assumed to be 0.
    norm : int
        Normalization constant.
    """
    
    return np.intersect1d(t-1, t).size / norm

# ======= #
# Classes #
# ======= #
class NThreshold():
    def __init__(self, n, params=None, rng=None):
        """Spin with a bias for activation and activation based on neighbors in
        previous time step.

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
        return -self.params[0]*s[:,0] - self.params[1]*s[:,1]

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """

        return np.array([self.p[self.n:].sum(),
                         self.p.dot(np.concatenate((np.arange(self.n), np.arange(self.n))))])
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
              K_cost=(0,50)):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(2)
        max_param_value : float, 20
        K_cost : twople, (0,50)
            (mean, std) of coupling cost

        Returns
        -------
        dict
        """

        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            return (np.linalg.norm(self.calc_observables() - constraints) +
                    abs(K_cost[0]-new_params[1])/K_cost[1])
        return minimize(cost, original_guess,
                        bounds=[(-max_param_value, max_param_value)]*2)
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin. Second col is number of active neighbors.
        """

        return np.vstack(([0]*self.n + [1]*self.n, list(range(self.n))*2)).T
#end NThreshold



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
        """Re-defines self.logZ and self.p using given parameters. These are the log
        partition function and the probability distribution over all configurations
        defined in the model.
        
        Parameters
        ----------
        params : ndarray
        """

        self.params = params
        
        self.logZ = fast_logsumexp(-self.calc_e(self.all_states(), self.params))[0]
        self.p = np.exp(-self.calc_e(self.all_states(), self.params) - self.logZ)
        
    def calc_e(self, s, params):
        return -s[:,0] * (self.params[0]*s[:,1] + self.params[1]*s[:,2])

    def calc_observables(self):
        """Calculate ensemble averaged observables using the probability distribution.

        Returns
        -------
        np.ndarray
            Consists of two elements (activation probability of center spin, typical
            number of active neighbors)
        """
        
        # b/c we know the ordering of the states and which contribute 0 to the observable, there is no need to
        # iterate explicitly over them
        return np.array([self.p[3*self.n:].sum(),
                         (self.p[2*self.n:3*self.n].dot(np.arange(self.n)) +
                          self.p[3*self.n:4*self.n].dot(np.arange(self.n)))])
    
    def si(self):
        """Active probability for center spin.

        Returns
        -------
        float
        """

        return self.p[self.n*2:].sum()

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
            return self.all_states()[self.rng.choice(4*self.n, p=self.p)][None,:]
        return self.all_states()[self.rng.choice(4*self.n, p=self.p, size=size)]
    
    def solve(self, constraints,
              original_guess=np.zeros(2),
              max_param_value=20,
              K_cost=(0,50),
              set_params=True):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(2)
        max_param_value : float, 20
        K_cost : twople, (0,50)
            (mean, std) of coupling cost
        set_params : bool, True

        Returns
        -------
        dict
        """

        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            return (np.linalg.norm(self.calc_observables() - constraints) +
                    abs(K_cost[0] - new_params[1])/K_cost[1])
        sol = minimize(cost, original_guess,
                       bounds=[(-max_param_value, max_param_value)]*2)

        if set_params:
            self.set_params(sol['x'])
        return sol
    
    def up_states(self):
        """All possible configurations in this model given that the center spin at
        t+1 is held at 1.
        
        Returns
        -------
        ndarray
            First col is center spin. Second col is number of active neighbors. First
            set of states correspond to setting center spin to 0, and second set when
            it is 1.
        """

        return np.vstack(([1]*(self.n*2), [0]*self.n + [1]*self.n, list(range(self.n)) + list(range(self.n)))).T

    def all_states(self):
        """All possible configurations in this model.
        
        Returns
        -------
        ndarray
            First col is center spin at t+1. Second col center spin at t. Third col
            is number of active neighbors. First set of states correspond to setting
            center spin to 0, and second set when it is 1.
        """

        inner_states = np.vstack(([0]*self.n + [1]*self.n, list(range(self.n)) + list(range(self.n))))
        return np.vstack(([0]*(2*self.n) + [1]*(2*self.n), np.hstack((inner_states, inner_states)))).T
#end NActivationIsing



class MarkovSimulator():
    def __init__(self, dtdx, polygons, gridix=0, rng=None):
        """
        Parameters
        ----------
        dtdx : twople
        polygons : gpd.GeoDataFrame
        gridix : int, 0
        rng : np.RandomState, None
        """
        
        assert 'model' in polygons.columns
        
        self.dtdx = dtdx
        self.polygons = polygons
        
        self.conf_df = discretize_conflict_events(*dtdx, gridix=gridix)
        self.rng = rng if not rng is None else np.random
       
    def simulate_NThreshold(self, T, save_every=1):
        """Simulate time series with single-step Markov chain using the 'model' column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        """

        polygons = self.polygons
        
        s = dict(zip(polygons.index, [0]*len(polygons)))
        history = np.zeros((T//save_every, len(polygons)), dtype=int)

        def new_state(poly):
            # use the number of active neighbors to condition on whether or not site is active
            n_active = len([True for n in poly['active_neighbors'] if s[n]])
            p_active = poly['model'].p[poly['n']+n_active]
            p_inactive = poly['model'].p[n_active]
            p_active /= p_active + p_inactive

            if np.random.rand() < p_active:
                return 1
            else:
                return 0

        polygons['s'] = np.zeros(len(polygons), dtype=int)
        for t in range(T):
            # for each cell, iterate it one time step
            polygons['s'] = polygons.apply(new_state, axis=1)

            if (t%save_every)==0:
                history[t//save_every,:] = polygons['s'].values

        self.history = pd.DataFrame(history, columns=polygons.index)

    def simulate_NActivationIsing(self, T, save_every=1):
        """Simulate time series with single-step Markov chain using the 'model' column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        """

        polygons = self.polygons  # DataFrame of Voronoi cells
        
        s = dict(zip(polygons.index, [0]*len(polygons)))  # current state of each polygon as {0,1}
        history = np.zeros((T//save_every, len(polygons)), dtype=int)  # save history to return

        def new_state(poly):
            # use the number of active neighbors to condition on whether or not site is active
            n_active = len([True for n in poly['active_neighbors'] if s[n]])
            if not poly['s']:  # when center spin is 0 in previous time step
                p_inactive = poly['model'].p[n_active]
                p_active = poly['model'].p[poly['n']*2+n_active]
            else:  # when center spin is 1 in previous time step
                p_inactive = poly['model'].p[poly['n']+n_active]
                p_active = poly['model'].p[poly['n']*3+n_active]

            p_active /= p_active + p_inactive

            if self.rng.rand() < p_active:
                return 1
            else:
                return 0

        polygons['s'] = np.zeros(len(polygons), dtype=int)
        for t in range(T):
            # for each cell, iterate it one time step
            polygons['s'] = polygons.apply(new_state, axis=1)

            if (t % save_every)==0:
                history[t//save_every,:] = polygons['s'].values

        self.history = pd.DataFrame(history, columns=polygons.index)

    def calc_pij(self):
        """Pair correlation between adjacent sites (t,t).

        Returns
        -------
        dict
            Key is ordered pair and value is correlation between conflict incidence
            (1) or non-incidence (0).
        """

        self.polygons['index'] = self.polygons.index
        tmn = self.conf_df['t'].min()
        tmx = self.conf_df['t'].max()

        # for every pair of neighbors, count how many times they appear together
        g_by_x = self.conf_df.groupby('x')
        def _pair_corr(poly):
            i = poly['index']
            pij = []
            time_series = np.zeros((tmx-tmn+1, 2), dtype=int)
            time_series[g_by_x.get_group(poly['index'])['t'].unique()-tmn,0] = 1

            # we could also check inactive neighbors
            for j in poly['active_neighbors']:
                time_series[:,1] = 0  # reset
                time_series[g_by_x.get_group(j)['t'].unique()-tmn,1] = 1
                pij.append((frozenset((i,j)), np.prod(time_series,axis=1).mean()))
            return pij

        pij = self.polygons.apply(_pair_corr, axis=1)
        pij = dict(chain.from_iterable(pij.values))
        return pij

    def calc_pij_delay(self, dt=1):
        """Pair correlation (i at t, and j at t-1).
        
        Parameters
        ----------
        dt : int, 1

        Returns
        -------
        dict
            Key is ordered pair and value is correlation between conflict incidence
            (1) or non-incidence (0).
        """
        
        assert dt>=1
        
        self.polygons['index'] = self.polygons.index
        tmx = self.conf_df['t'].max()
        tmn = self.conf_df['t'].min()

        # for every pair of neighbors, count how many times they appear together
        g_by_x = self.conf_df.groupby('x')
        def _pair_corr(poly):
            i = poly['index']
            pij = []
            time_series = np.zeros((tmx-tmn+1, 2), dtype=int)
            time_series[g_by_x.get_group(poly['index'])['t'].unique()-tmn,0] = 1

            # we could also check inactive neighbors
            for j in poly['active_neighbors']:
                time_series[:,1] = 0  # reset
                time_series[g_by_x.get_group(j)['t'].unique()-tmn,1] = 1
                pij.append(((i,j), (time_series[dt:,0]*time_series[:-dt,1]).mean()))
            return pij

        pij_delay = self.polygons.apply(_pair_corr, axis=1)
        pij_delay = dict(chain.from_iterable(pij_delay.values))
        return pij_delay
#end MarkovSimulator

