# ====================================================================================== #
# Simulating and solving for models of conflict time series.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from scipy.optimize import minimize
from coniii.enumerate import fast_logsumexp
from coniii.utils import bin_states
import swifter 
from swifter import set_defaults

from .construct import discretize_conflict_events
from .utils import *


set_defaults(
    npartitions=None,
    dask_threshold=1,
    scheduler="processes",
    progress_bar=False,
    progress_bar_desc=None,
    allow_dask_on_strings=False,
    force_parallel=False,
)

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

def solve_delayed_activation(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=1):
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
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')  # all conflicts grouped by time bin
    g_by_t.get_group(conf_df['t'].iloc[0]);  # cache groups
    if n_cpus==1:
        polygons['params'], polygons['constraints'] = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            polygons['params'], polygons['constraints'] = list(zip(*pool.map(loop_wrapper, polygons.index,
                                                                             chunksize=20)))

    # read out parmeters into separate cols
    polygons['h'] = polygons['params'].apply(lambda i:i[0])
    polygons['J'] = polygons['params'].apply(lambda i:i[1])

    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NActivationIsing(i['n'], params=np.array([i['params'][0], i['params'][1]]))
                         for _,i in polygons.iterrows()]

def solve_NThreshold1(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=None):
    """Solve for the parameters of a delayed activation NThreshold model given the data.
    Model solutions saved into polygons DataFrame.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'K'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
        Inverse weight for prior on coupling K.
    n_cpus : int, None
        Multiprocess unless this is 1.
    """
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / (tmx-tmn+1)

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
        pin /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NThreshold1(n)
        constraints = np.array([pi, pin])
        solver.solve(constraints, max_param_value=10, K_cost=(0, K_cost_inverse_weight))
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(conf_df['t'].iloc[0]);  # cache to memory

    if n_cpus==1:
        polygons['params'], polygons['constraints'] = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            polygons['params'], polygons['constraints'] = list(zip(*pool.map(loop_wrapper, polygons.index,
                                                                             chunksize=20)))

    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NThreshold1(i['n'], params=np.array([i['params'][0], i['params'][1]]))
                         for _,i in polygons.iterrows()]

def solve_NThreshold2(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=None):
    """Solve for the parameters of a delayed activation NThreshold model given the data.
    Model solutions saved into polygons DataFrame.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'K'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
        Inverse weight for prior on coupling K.
    n_cpus : int, None
        Multiprocess unless this is 1.
    """
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / (tmx-tmn+1)

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """

        pi = px.loc[i]
        pin = 0

        # for each time point, count number of neighbors at t-1 only if the center
        # site is active
        for t, g in g_by_t:
            n_active = 0
            if i in g['x'].values and t-1 in g_by_t.groups.keys():
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        n_active += 1
            pin += n_active
        pin /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NThreshold2(n)
        constraints = np.array([pi, pin])
        sol = solver.solve(constraints, max_param_value=20, K_cost=(0, K_cost_inverse_weight))
        solver.set_params(sol['x'])
        # when sol is non-convergent, try Powell algorithm
        if np.linalg.norm(solver.calc_observables() - constraints) > 1e-3:
            sol = solver.solve(constraints,
                               max_param_value=20,
                               K_cost=(0, K_cost_inverse_weight),
                               method='powell')
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(conf_df['t'].iloc[0]);  # cache to memory
    if n_cpus==1:
        polygons['params'], polygons['constraints'] = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            polygons['params'], polygons['constraints'] = list(zip(*pool.map(loop_wrapper, polygons.index,
                                                                             chunksize=20)))
    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NThreshold2(i['n'], params=np.array([i['params'][0], i['params'][1]]))
                         for _,i in polygons.iterrows()]
    return polygons

def solve_NThreshold3(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=None):
    """Solve for the parameters of a delayed activation NThreshold3 model given the
    data.  Model solutions saved into polygons DataFrame.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'K'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
        Inverse weight for prior on coupling K.
    n_cpus : int, None
        Multiprocess unless this is 1.
    """
    
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / (tmx-tmn+1)

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """

        pi = px.loc[i]
        pin = 0

        # for each time point, count number of neighbors at t-1 only if the center
        # site is active
        for t, g in g_by_t:
            n_active = 0
            if i in g['x'].values and t-1 in g_by_t.groups.keys():
                # check if center site was in the past at t-1
                if i in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                    n_active += 1
                # check if each neighbor site j was in the past at t-1
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        n_active += 1
            pin += n_active
        pin /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NThreshold3(n)
        constraints = np.array([pi, pin])
        solver.solve(constraints, max_param_value=10, K_cost=(0, K_cost_inverse_weight))
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(conf_df['t'].iloc[0]);  # cache to memory
    if n_cpus==1:
        polygons['params'], polygons['constraints'] = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            polygons['params'], polygons['constraints'] = list(zip(*pool.map(loop_wrapper, polygons.index,
                                                                             chunksize=20)))
    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NThreshold3(i['n'], params=np.array([i['params'][0], i['params'][1]]))
                         for _,i in polygons.iterrows()]

def solve_NThreshold4(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=None):
    """Solve for the parameters of a delayed activation NThreshold4 model given the
    data.  Model solutions saved into polygons DataFrame.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'K'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
        Inverse weight for prior on coupling K.
    n_cpus : int, None
        Multiprocess unless this is 1.
    """
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / (tmx-tmn+1)

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """
        pi = px.loc[i]
        pii = 0
        pin = 0

        # for each time point, count number of neighbors at t-1 only if the center
        # site is active
        for t, g in g_by_t:
            i_active = 0
            n_active = 0
            if i in g['x'].values and t-1 in g_by_t.groups.keys():
                # check if self was active in past t-1
                if i in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                    i_active += 1

                # check if each neighbor site j was in the past at t-1
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        n_active += 1
            pii += i_active
            pin += n_active
        pii /= tmx - tmn
        pin /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NThreshold4(n)
        constraints = np.array([pi, pii, pin])
        sol = solver.solve(constraints, max_param_value=10, K_cost=(0, K_cost_inverse_weight))
        solver.set_params(sol['x'])
        # when sol is non-convergent, try Powell algorithm
        if np.linalg.norm(solver.calc_observables() - constraints) > 1e-3:
            sol = solver.solve(constraints,
                               max_param_value=20,
                               K_cost=(0, K_cost_inverse_weight),
                               method='powell')
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(conf_df['t'].iloc[0])  # cache to memory
    if n_cpus==1:
        output = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            output = list(zip(*pool.map(loop_wrapper, polygons.index, chunksize=20)))
    polygons['params'] = output[0]
    polygons['constraints'] = output[1]

    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NThreshold4(i['n'],
                                     params=np.array([i['params'][0], i['params'][1], i['params'][2]]))
                         for _,i in polygons.iterrows()]
    return polygons

def solve_NThreshold5(polygons, conf_df, K_cost_inverse_weight=10, n_cpus=None):
    """Solve for the parameters of a delayed activation NThreshold5 model given the
    data.  Model solutions saved into polygons DataFrame.

    Parameters
    ----------
    polygons : pd.DataFrame
        Updated in place with parameters col 'params' and separately the bias
        parameter 'h' and activation parameter 'K'.
    conf_df : pd.DataFrame
        Conflict events.
    K_cost_inverse_weight : float, 10
        Inverse weight for prior on coupling K.
    n_cpus : int, None
        Multiprocess unless this is 1.
    """
    tmn = conf_df['t'].min()
    tmx = conf_df['t'].max()
    px = conf_df.groupby('x')['t'].unique().apply(lambda i:len(i)) / (tmx-tmn+1)

    def loop_wrapper(i):
        """For a given site, get it's typical activation probability along with
        its pair correlations no. of active neighbors.
        """
        pi = px.loc[i]
        pii = 0
        pij = dict([(j,0) for j in polygons['active_neighbors'].loc[i]])

        # for each time point, count number of neighbors at t-1 only if the center
        # site is active
        for t, g in g_by_t:
            i_active = 0
            n_active = 0
            if i in g['x'].values and t-1 in g_by_t.groups.keys():
                # check if self was active in past t-1
                if i in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                    pii += 1

                # check if each neighbor site j was in the past at t-1
                for j in polygons['active_neighbors'].loc[i]:
                    if j in conf_df['x'].loc[g_by_t.groups[t-1]].values:
                        pij[j] += 1
        pii /= tmx - tmn
        for j in pij.keys():
            pij[j] /= tmx - tmn
        n = len(polygons['active_neighbors'].loc[i]) + 1

        # must convert pairwise correlations to {-1,1} basis
        solver = NThreshold5(n)
        constraints = np.concatenate(([pi, pii],list(pij.values())))
        sol = solver.solve(constraints, K_cost=(0, K_cost_inverse_weight))
        solver.set_params(sol['x'])
        # when sol is non-convergent, try Powell algorithm
        if np.linalg.norm(solver.calc_observables() - constraints) > 1e-3:
            sol = solver.solve(constraints,
                               max_param_value=20,
                               K_cost=(0, K_cost_inverse_weight),
                               method='powell')
        return solver.params, constraints

    g_by_t = conf_df.groupby('t')
    g_by_t.get_group(conf_df['t'].iloc[0])  # cache to memory
    if n_cpus==1:
        output = list(zip(*[loop_wrapper(i) for i in polygons.index]))
    else:
        with Pool(n_cpus) as pool:
            output = list(zip(*pool.map(loop_wrapper, polygons.index, chunksize=20)))
    polygons['params'] = output[0]
    polygons['constraints'] = output[1]

    polygons['n'] = polygons['active_neighbors'].apply(lambda i: len(i)+1)
    polygons['model'] = [NThreshold5(i['n'], params=i['params'])
                         for _,i in polygons.iterrows()]
    return polygons

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
        
    def si(self):
        """Activation probability of center spin."""

        return self.p[self.n:].sum()

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
              original_guess=[-1,0],
              max_param_value=20,
              K_cost=(0,50),
              **solver_kw):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(2)
        max_param_value : float, 20
        K_cost : twople, (0,50)
            (mean, std) of coupling cost
        **solver_kw

        Returns
        -------
        dict
        """

        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            cost = (np.linalg.norm(self.calc_observables() - constraints) +
                    abs(K_cost[0]-new_params[1])/K_cost[1])
            return cost
        return minimize(cost, original_guess,
                        bounds=[(-max_param_value, max_param_value)]*2,
                        **solver_kw)
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin. Second col is number of active neighbors.
        """

        return np.vstack(([0]*self.n + [1]*self.n, list(range(self.n))*2)).T
    
    def __eq__(self, other):
        return np.array_equal(self.params, other.params)

    def __lt__(self, other):
        return np.linalg.norm(self.params) < np.linalg.norm(other.params)

    def __gt__(self, other):
        return np.linalg.norm(self.params) > np.linalg.norm(other.params)
#end NThreshold



class NThreshold1(NThreshold):
    def calc_e(self, s, params):
        return -self.params[0]*s[:,0] - self.params[1]*s[:,1]

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """
        return np.array([self.p[self.n:].sum(),
                         self.p.dot(np.concatenate((np.arange(self.n), np.arange(self.n))))])
#end NThreshold1



class NThreshold2(NThreshold):
    def calc_e(self, s, params):
        return -self.params[0]*s[:,0] - self.params[1]*s[:,0]*s[:,1]

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """
        return np.array([self.p[self.n:].sum(),
                         self.p[self.n:].dot(np.arange(self.n))])
#end NThreshold2



class NThreshold3(NThreshold):
    # remember self.n is 1+no_of_neighbors, so the full set of possible "neighbors"
    # in the past is self.n+1
    def si(self):
        """Activation probability of center spin."""
        return self.p[self.n+1:].sum()

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
            return self.all_states()[self.rng.choice(2*self.n+2, p=self.p)][None,:]
        return self.all_states()[self.rng.choice(2*self.n+2, p=self.p, size=size)]
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin at t+1. Second col is number of active neighbors
            at t.
        """
        return np.vstack(([0]*(self.n+1) + [1]*(self.n+1), list(range(self.n+1))*2)).T

    def calc_e(self, s, params):
        return -self.params[0]*s[:,0] - self.params[1]*s[:,1]

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """
        return np.array([self.p[self.n+1:].sum(),
                         self.p.dot(np.concatenate((np.arange(self.n+1), np.arange(self.n+1))))])
#end NThreshold3



class NThreshold4(NThreshold):
    """Model where there is a separate interaction for t+1 with the center site at t
    and the number neighbors. There is also a separate bias.
    """
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
            assert params.size==3 and params.ndim==1
            self.set_params(params)
        else:
            self.set_params(np.zeros(3))
        self.rng = rng if not rng is None else np.random
 
    # remember self.n is 1+no_of_neighbors
    def si(self):
        """Activation probability of center spin in the future."""
        return self.p[2*self.n:].sum()

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
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin at t+1. Second col is self in the past. Third
            col is the number of active neighbors at t.
        """
        return np.vstack(([0]*(2*self.n) + [1]*(2*self.n), 
                          [0]*self.n + [1]*self.n + [0]*self.n + [1]*self.n,
                          list(range(self.n))*4)).T

    def calc_e(self, s, params):
        return -s[:,0] * (self.params[0] + self.params[1]*s[:,1] + self.params[2]*s[:,2])

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """
        return np.array([self.p[2*self.n:].sum(),
                         self.p[3*self.n:].sum(),
                         self.p[2*self.n:].dot(np.concatenate((np.arange(self.n), np.arange(self.n))))])

    def solve(self, constraints,
              original_guess=[-1,0,0],
              max_param_value=20,
              K_cost=(0,50),
              **solver_kw):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, [-1,0,0]
        max_param_value : float, 20
        K_cost : twople, (0,50)
            (mean, std) of coupling cost
        **solver_kw

        Returns
        -------
        dict
        """
        assert max_param_value > 0
        
        def cost(new_params):
            self.set_params(new_params)
            cost = (np.linalg.norm(self.calc_observables() - constraints) +
                    abs(K_cost[0]-new_params[2])/K_cost[1])
            return cost
        return minimize(cost, original_guess,
                        bounds=[(-max_param_value, max_param_value)]*3,
                        **solver_kw)
#end NThreshold4



class NThreshold5(NThreshold):
    """Model where there is a separate interaction for t+1 with the center site at t
    and each possible neighbor. There is also a separate bias for activity.
    """
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
        assert n<10, "Not designed for large systems."
        self.n = n
        if not params is None:
            assert params.size==(n+1) and params.ndim==1
            self.set_params(params)
        else:
            self.set_params(np.zeros(n+1))
        self.rng = rng if not rng is None else np.random
 
    # remember self.n is 1+no_of_neighbors
    def si(self):
        """Activation probability of center spin in the future."""
        return self.p[2**self.n:].sum()

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
            return self.all_states()[self.rng.choice(2**self.n, p=self.p)][None,:]
        return self.all_states()[self.rng.choice(2**self.n, p=self.p, size=size)]
    
    def all_states(self):
        """All possible configuration in this model.
        
        Returns
        -------
        ndarray
            First col is center spin at t+1. Second col is self in the past.
            Remaining cols are neighbors.
        """
        return bin_states(self.n+1)

    def calc_e(self, s, params):
        #return -(self.params[0] + (s[:,1:]*2.-1).dot(self.params[1:])) * (s[:,0]*2.-1)
        return -(self.params[0] + s[:,1:].dot(self.params[1:])) * s[:,0]

    def calc_observables(self):
        """Typical activation prob and typical no. of neighbors active in previous
        time step.
        """
        n = self.n
        all_states = self.all_states()
        return np.insert(((self.p[2**(n-1):]*all_states[2**(n-1):,0])[:,None]*all_states[2**(n-1):,1:]).sum(0),
                         0,
                         self.si())

    def solve(self, constraints,
              original_guess=None,
              max_param_value=20,
              K_cost=(0,50),
              **solver_kw):
        """
        Parameters
        ----------
        constraints : ndarray
        original_guess : ndarray, np.zeros(self.n)
        max_param_value : float, 20
        K_cost : twople, (0,50)
            (mean, std) of coupling cost
        **solver_kw

        Returns
        -------
        dict
        """
        assert max_param_value > 0
        original_guess = np.zeros(self.n+1) if original_guess is None else original_guess
        assert len(original_guess)==(self.n+1)
        
        def cost(new_params):
            self.set_params(new_params)
            cost = (np.linalg.norm(self.calc_observables() - constraints) +
                    (np.abs(K_cost[0]-new_params[2:])/K_cost[1]).sum())
            return cost
        return minimize(cost, original_guess,
                        bounds=[(-max_param_value, max_param_value)]*(self.n+1),
                        **solver_kw)
#end NThreshold5




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
        # b/c we know the ordering of the states and which contribute 0 to the observable,
        # there is no need to iterate explicitly over them
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
       
    def simulate_NThreshold(self, T, save_every=1, s0=None):
        """Simulate time series with single-step Markov chain using the 'model'
        column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        s0 : ndarray or list
            Initial state at which to start simulation.
        """
        polygons = self.polygons
        if s0 is None:
            s0 = [0]*len(polygons)
        else:
            assert set(s0) <= frozenset((0,1))
        s = dict(zip(polygons.index, s0))  # current state of each polygon as {0,1}
        new_s = dict(zip(polygons.index, [0]*len(polygons)))  # next state of each polygon

        # read in cols of polygons for use in faster loop
        neighbors = dict(polygons['active_neighbors'])
        model = dict(polygons['model'])
        n = dict(polygons['n'])
        history = []

        def new_state(i):
            # probability of a state being active depends on the no. of active
            # neighbors; once that is fixed then there are two possibilities, it is up or
            # it is down and these must be normalized to unity
            n_active = len([True for n in neighbors[i] if s[n]])
            if isinstance(model[i], NThreshold1) or isinstance(model[i], NThreshold2):
                p_active = model[i].p[n[i]+n_active]
                p_inactive = model[i].p[n_active]
            elif isinstance(model[i], NThreshold4):
                self_active = s[i]==1
                if self_active:  # when center spin is active in past
                    p_active = model[i].p[3*n[i]+n_active]
                    p_inactive = model[i].p[n[i]+n_active]
                else:
                    p_active = model[i].p[2*n[i]+n_active]
                    p_inactive = model[i].p[n_active]
            else:
                raise NotImplementedError
            p_active /= p_active + p_inactive

            if np.random.rand() < p_active:
                return 1
            return 0

        for t in range(T):
            # for each cell, iterate it one time step
            for i in polygons.index:
                new_s[i] = new_state(i)

            if (t%save_every)==0:
                history.append(list(s.values()))
            s = new_s.copy()

        self.history = pd.DataFrame(history, columns=polygons.index)

    def simulate_NThreshold5(self, T, save_every=1, s0=None):
        """Simulate time series with single-step Markov chain using the 'model'
        column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        s0 : ndarray or list
            Initial state at which to start simulation.
        """
        polygons = self.polygons
        if s0 is None:
            s0 = [0]*len(polygons)
        else:
            assert set(s0) <= frozenset((0,1))
        s = dict(zip(polygons.index, s0))  # current state of each polygon as {0,1}
        new_s = dict(zip(polygons.index, [0]*len(polygons)))  # next state of each polygon

        # read in cols of polygons for use in faster loop
        neighbors = dict(polygons['active_neighbors'])
        model = dict(polygons['model'])
        n = dict(polygons['n'])
        history = []

        def new_state(i):
            # probability of a state being active depends on the no. of active
            # neighbors; once that is fixed then there are two possibilities, it is up or
            # it is down and these must be normalized to unity
            n_active = np.array([True if s[n] else False for n in neighbors[i]])
            # this should give the relative index accounting for the neighbors
            counter = int(n_active.dot(2**np.arange(len(n_active), dtype=int)[::-1]).sum())
            if s[i]:
                p_inactive = model[i].p[2**(n[i]-1)+counter]
                p_active = model[i].p[2**(n[i]-1)*3+counter]
            else:
                p_inactive = model[i].p[counter]
                p_active = model[i].p[2**n[i]+counter]
            p_active /= p_active + p_inactive

            if np.random.rand() < p_active:
                return 1
            return 0

        for t in range(T):
            # for each cell, iterate it one time step
            for i in polygons.index:
                new_s[i] = new_state(i)

            if (t%save_every)==0:
                history.append(list(s.values()))
            s = new_s.copy()

        self.history = pd.DataFrame(history, columns=polygons.index)

    def simulate_NThreshold2(self, T, save_every=1, s0=None):
        """Simulate time series with single-step Markov chain using the 'model'
        column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        s0 : ndarray or list
            Initial state at which to start simulation.
        """
        
        polygons = self.polygons
        if s0 is None:
            s0 = [0]*len(polygons)
        else:
            assert set(s0) <= frozenset((0,1))
        s = dict(zip(polygons.index, s0))  # current state of each polygon as {0,1}
        new_s = dict(zip(polygons.index, [0]*len(polygons)))  # next state of each polygon

        # read in cols of polygons for use in faster loop
        neighbors = dict(polygons['active_neighbors'])
        model = dict(polygons['model'])
        n = dict(polygons['n'])
        history = []

        def new_state(i):
            # probability of a state being active depends on the no. of active
            # neighbors; once that is fixed then there are two possibilities, it is up or
            # it is down and these must be normalized to unity
            n_active = len([True for n in neighbors[i] if s[n]])
            p_active = model[i].p[n[i]+n_active]
            p_inactive = model[i].p[n_active]
            p_active /= p_active + p_inactive

            if np.random.rand() < p_active:
                return 1
            return 0

        for t in range(T):
            # for each cell, iterate it one time step
            for i in polygons.index:
                new_s[i] = new_state(i)

            if (t%save_every)==0:
                history.append(list(s.values()))
            s = new_s.copy()

        self.history = pd.DataFrame(history, columns=polygons.index)

    def simulate_NThreshold3(self, T, save_every=1, s0=None):
        """Simulate time series with single-step Markov chain using the 'model'
        column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        s0 : ndarray or list
            Initial state at which to start simulation.
        """
        
        polygons = self.polygons
        if s0 is None:
            s0 = [0]*len(polygons)
        else:
            assert set(s0) <= frozenset((0,1))
        s = dict(zip(polygons.index, s0))  # current state of each polygon as {0,1}
        new_s = dict(zip(polygons.index, [0]*len(polygons)))  # next state of each polygon

        # read in cols of polygons for use in faster loop
        neighbors = dict(polygons['active_neighbors'])
        model = dict(polygons['model'])
        n = dict(polygons['n'])
        history = []

        def new_state(i):
            # probability of a state being active depends on the no. of active
            # neighbors (can include self); once that is fixed then there are two
            # possibilities, it is up or it is down and these must be normalized to
            # unity
            n_active = len([True for n in neighbors[i] if s[n]])
            if s[i]: n_active += 1
            p_active = model[i].p[n[i]+n_active]
            p_inactive = model[i].p[n_active]
            p_active /= p_active + p_inactive

            if np.random.rand() < p_active:
                return 1
            return 0

        for t in range(T):
            # for each cell, iterate it one time step
            for i in polygons.index:
                new_s[i] = new_state(i)

            if (t%save_every)==0:
                history.append(list(s.values()))
            s = new_s.copy()

        self.history = pd.DataFrame(history, columns=polygons.index)

    def simulate_NActivationIsing(self, T, save_every=1, p_up=0):
        """Simulate time series with single-step Markov chain using the 'model'
        column in polygons.

        Parameters
        ----------
        T : int
        save_every : int, 1
        p_up : float, 0.
            Probability of a spin being up in initial configuration.
        """

        polygons = self.polygons  # DataFrame of Voronoi cells
        
        s = dict(zip(polygons.index, [0 if np.random.rand()<p_up else 1
                                      for i in range(len(polygons))]))  # current state of each polygon as {0,1}
        new_s = dict(zip(polygons.index, [0]*len(polygons)))  # next state of each polygon as {0,1}

        # read in cols of polygons for use in faster loop
        neighbors = dict(polygons['active_neighbors'])
        model = dict(polygons['model'])
        n = dict(polygons['n'])
        history = []  # save history to return

        def new_state(i):
            # use the number of active neighbors to condition on whether or not site is active
            n_active = len([True for n in neighbors[i] if s[n]])
            if not s[i]:  # when center spin is 0 in previous time step
                p_inactive = model[i].p[n_active]
                p_active = model[i].p[n[i]*2+n_active]
            else:  # when center spin is 1 in previous time step
                p_inactive = model[i].p[n[i]+n_active]
                p_active = model[i].p[n[i]*3+n_active]

            p_active /= p_active + p_inactive

            if self.rng.rand() < p_active:
                return 1
            else:
                return 0

        for t in range(T):
            # for each cell, iterate it one time step
            for i in polygons.index:
                new_s[i] = new_state(i)

            if (t % save_every)==0:
                history.append(list(s.values()))
            s = new_s.copy()

        self.history = pd.DataFrame(history, columns=polygons.index)

    def calc_pij(self):
        """Pair correlation between adjacent sites at same time t.

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

            # we could also check inactive neighbors, but they are not checked below
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

