# ====================================================================================== #
# Module for construction conflict avalanches such as discretizing conflicts to spatial
# and temporal bins and connecting them to one another.
# Author: Eddie Lee, Niraj Kushwaha
# ====================================================================================== #
from voronoi_globe.interface import load_voronoi
from shapely.geometry import Point
import swifter
from functools import cache
import warnings
import itertools

from .network import CausalGraph
from .utils import *
from .data import ACLED2020



class Avalanche():
    """For constructing causal avalanches.
    """
    def __init__(self, dt, dx, gridix=0, conflict_type='battles',
                 sig_threshold=95,
                 rng=None,
                 iprint=False):
        assert 0<=sig_threshold<100

        self.dt = dt
        self.dx = dx
        self.gridix = gridix
        self.conflict_type = conflict_type
        self.sig_threshold = sig_threshold
        self.rng = rng or np.random
        self.iprint = iprint
        
        self.polygons = load_voronoi(dx, gridix)
        self.time_series = discretize_conflict_events(dt, dx, gridix, conflict_type)[['t','x']]
        self.setup_causal_graph()
        if self.iprint: print("Starting avalanche construction...")
        self.construct()
    
    def setup_causal_graph(self, shuffles=100):
        """Calculate transfer entropy between neighboring polygons and save it as a
        causal network graph.

        Parameters
        ----------
        shuffles : int, 100
        """
        
        tmx = self.time_series['t'].max()
        self_edges = {}
        pair_edges = {}
            
        # self edges
        def loop_wrapper(args):
            self.rng.seed()

            x, time_series = args
            vals = [self_te(np.unique(time_series), tmx),[]]
            for i in range(shuffles):
                vals[1].append(self_te(np.unique(self.rng.randint(tmx+1, size=time_series.size)), tmx))
            return x, vals

        with Pool() as pool:
            self_edges = dict(pool.map(loop_wrapper, self.time_series.groupby('x')['t']))
        if self.iprint: print("Done with self edges.")

        # cell 2 cell edges
        group = self.time_series.groupby('x')['t']
        def loop_wrapper(args):
            pair_edges = []
            x, time_series = args
            neighbors = self.polygons['neighbors'].loc[x]
            for n in neighbors:
                if n in self.time_series['x'].values:
                    n_time_series = group.get_group(x)
                    vals = [pair_te(np.unique(time_series), np.unique(n_time_series), tmx),[]]

                    for i in range(shuffles):
                        # randomly place each event in any time bin
                        time_series = self.rng.randint(tmx+1, size=time_series.size)
                        n_time_series = self.rng.randint(tmx+1, size=n_time_series.size)
                        vals[1].append(pair_te(np.unique(time_series), np.unique(n_time_series), tmx))
                    pair_edges.append([(x,n), vals])
            return pair_edges

        with Pool() as pool:
            pair_edges = dict(list(itertools.chain.from_iterable(pool.map(loop_wrapper, group))))
        if self.iprint: print("Done with pair edges.")

        self.causal_graph = CausalGraph(sig_threshold=self.sig_threshold)
        self.causal_graph.setup(self_edges, pair_edges)

    def construct(self):
        """Construct causal avalanches of conflict events."""

        ava = []  # indices of conflict events
        remaining_ix = set(self.time_series.index)
        to_check = set()
        checked = set()
        time_group = self.time_series.groupby('t')

        while remaining_ix:
            ix = remaining_ix.pop()
            to_check.add(ix)
            ava.append([])
            while to_check:
                ava[-1].append(to_check.pop())
                checked.add(ava[-1][-1])
                start_ix = ava[-1][-1]
                t = self.time_series['t'].loc[ava[-1][-1]]

                # add successors which must be at the next time step
                if t+1 in time_group.groups.keys():
                    df = time_group.get_group(t+1) 
                    for n in self.causal_graph.neighbors(self.time_series['x'].loc[ava[-1][-1]]):
                        ix = n==df['x']
                        if ix.any():
                            # remove events from being added to another avalanches
                            for i in df.index[ix]:
                                try:
                                    remaining_ix.remove(i)
                                except:
                                    pass
                            # add them to the current avalanche
                            ava[-1].extend(df.index[ix])
                            [checked.add(i) for i in df.index[ix]]
                            # make sure they will be checked themselves for neighbors
                            if not ava[-1][-1] in checked:
                                to_check.add(ava[-1][-1])

                # add predecessors which must be at the previous time step
                if t-1 in time_group.groups.keys():
                    df = time_group.get_group(t-1) 
                    for n in self.causal_graph.predecessors(self.time_series['x'].loc[ava[-1][-1]]):
                        ix = n==df['x']
                        if ix.any():
                            # remove events from being added to another avalanches
                            for i in df.index[ix]:
                                try:
                                    remaining_ix.remove(i)
                                except:
                                    pass
                            # add them to the current avalanche
                            ava[-1].extend(df.index[ix])
                            [checked.add(i) for i in df.index[ix]]
                            # make sure they will be checked themselves for neighbors
                            if not ava[-1][-1] in checked:
                                to_check.add(ava[-1][-1])

        # conflict avalanches, index is index of conflict event in conflict events DataFrame
        self.avalanches = ava
#end Avalanche



def pair_te(t1, t2, tmx):
    """Is t1 explained by t2 or just by itself?"""
    from entropy.entropy import joint_p_mat, bin_states

    X = np.zeros((tmx+1, 3), dtype=int)
    X[t1,0] = 1
    X[t1-1,1] = 1
    X[t2,2] = 1
    X = X[:-1]
    X = np.vstack((X, bin_states(3)))
    
    # probability distribution over all possible configurations
    # first col indicates how t1 lines up with itself over a delayed time interval
    # second col indicates dependency on t2
    pmat = np.unique(X, axis=0, return_counts=True)[1] - 1  # remove padded states
    pmat = np.vstack((pmat[::2], pmat[1::2])).T / pmat.sum()
    
    p_terms = np.array([[pmat[0,0], pmat[0].sum() * pmat[:,0].sum()],
                        [pmat[0,1], pmat[0].sum() * pmat[:,1].sum()],
                        [pmat[1,0], pmat[1].sum() * pmat[:,0].sum()],
                        [pmat[1,1], pmat[1].sum() * pmat[:,1].sum()],
                        [pmat[2,0], pmat[2].sum() * pmat[:,0].sum()],
                        [pmat[2,1], pmat[2].sum() * pmat[:,1].sum()],
                        [pmat[3,0], pmat[3].sum() * pmat[:,0].sum()],
                        [pmat[3,1], pmat[3].sum() * pmat[:,1].sum()]])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        te = np.nansum(p_terms[:,0] * (np.log(p_terms[:,0]) - np.log(p_terms[:,1])))

    return te

def self_te(t, tmx):
    """Self transfer entropy calculation only knowing time points at which events
    occurred. Naturally, this is only for binary time series.

    Parameters
    ----------
    t : ndarray
        Assuming only unique values.
    tmx : int

    Returns
    -------
    float
    """
    
    (p11, p01, p10, p00), (p1, p0) = _self_probabilities(t, tmx)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        te = np.nansum([p11 * np.log(p11/p1**2),
                        p10 * np.log(p10/p1/p0),
                        p01 * np.log(p01/p0/p1),
                        p00 * np.log(p00/p0**2)]) / np.log(2)
    return te

def _self_probabilities(t, tmx):
    """Helper function for self_edges()."""

    # the complement of event times t
    t_comp = np.delete(np.arange(tmx+1), t)
    
    # use intersections to count possible outcomes
    p11 = np.in1d(t+1, t, assume_unique=True).sum()
    p01 = np.in1d(t_comp+1, t, assume_unique=True).sum()
    p10 = np.in1d(t+1, t_comp, assume_unique=True).sum()
    p00 = tmx - (p11 + p01 + p10)
    assert p00>=0, p00
    
    norm = p11 + p01 + p10 + p00
    p11 /= norm
    p01 /= norm
    p10 /= norm
    p00 /= norm

    p1 = t.size / tmx
    p0 = 1-p1

    return (p11, p01, p10, p00), (p1, p0)

@cache
def discretize_conflict_events(dt, dx, gridix=0, conflict_type='battles'):
    """Merged GeoDataFrame for conflict events of a certain type into the Voronoi
    cells. Time discretized.

    Cached in order to save time for fine grid cells.

    Parameters
    ----------
    dt : int
    dx : int
    gridix : int, 0
    conflict_type : str, 'battles'

    Returns
    -------
    GeoDataFrame
        New columns 't' and 'x' indicate time and Voronoi bin indices.
    """
    
    polygons = load_voronoi(dx, gridix)
    df = ACLED2020.battles_df()
    conflict_ev = gpd.GeoDataFrame(df[['event_date','longitude','latitude']],
                                   geometry=gpd.points_from_xy(df.longitude,df.latitude),
                                   crs=polygons.crs)
    conflict_ev['t'] = (conflict_ev['event_date']-conflict_ev['event_date'].min()) // np.timedelta64(dt,'D')
    
    conflict_ev = gpd.sjoin(conflict_ev, polygons, how='left', op='within')
    conflict_ev.rename(columns={'index_right':'x'}, inplace=True)

    # no need for polygon neighbors column or raw index
    return conflict_ev.drop(['neighbors','index'], axis=1)

