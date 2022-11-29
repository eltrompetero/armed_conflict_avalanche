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

from .network import *
from .utils import *
from .data import ACLED2020
from .transfer_entropy_func import *
from .self_loop_entropy_func import *



class Avalanche():
    """For constructing causal avalanches.
    """
    def __init__(self, dt, dx,
                 gridix=0,
                 conflict_type='battles',
                 sig_threshold=95,
                 rng=None,
                 iprint=False,
                 setup=True,
                 shuffle_null=False):
        """
        Parameters
        ----------
        dt : int
            Time separation scale.
        dx : int
            Inverse distance separation scale.
        gridix : int, 0
            Random Voronoi grid index.
        conflict_type : str, 'battles'
        sig_threshold : int, 95
        rng : np.random.RandomState, None
        iprint : bool, False
        setup : bool, True
            If False, don't run causal graph and avalanche construction.
        shuffle_null : bool, False
        """

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
        self.time_series_CG_generator()
        if shuffle_null:
            if iprint: print("Starting shuffling...")
            self.randomize()

        if setup:
            self.setup_causal_graph()
            if self.iprint: print("Starting avalanche construction...")
            self.construct()
    
    def randomize(self):
        """Randomize time index in each polygon.
        """

        g_by_x = self.time_series.groupby('x')
        tmx = self.time_series['t'].max()

        randomized_time_series = []
        for x, thisg in g_by_x:
            # replace every occurrence of t with a random choice from all possible time bins
            uniqt = np.unique(thisg['t'])
            trand = self.rng.choice(np.arange(tmx+1), size=uniqt.size, replace=False)
            newt = np.zeros(thisg.shape[0], dtype=int)
            for t_, tr_ in zip(uniqt, trand):
                newt[thisg['t']==t_] = tr_
            thisg['t'].values[:] = newt
            randomized_time_series.append(thisg)

        randomized_time_series = pd.concat(randomized_time_series)
        self.time_series = randomized_time_series

    def setup_causal_graph(self, shuffles=100):
        """Calculate transfer entropy between neighboring polygons and save it as a
        causal network graph.

        Parameters
        ----------
        shuffles : int, 100
        """

        self_edges = self_links(self.time_series_CG_matrix, number_of_shuffles=shuffles)
        pair_edges = links(self.time_series_CG_matrix, self.polygons.drop('geometry' , axis=1), number_of_shuffles=shuffles)
        
        self.causal_graph = CausalGraph()
        self.causal_graph.setup(self_edges, pair_edges, sig_threshold=self.sig_threshold)

    def time_series_CG_generator(self):
        """Generates a coarse-grained (depending on dt and dx) time series matrix for the 
        whole dataset.
        """
        
        time_series_unique = self.time_series.drop_duplicates()
        
        col_labels = np.sort(np.unique(time_series_unique["x"].to_numpy()))
        time_series_CG_matrix = np.zeros((time_series_unique["t"].max()+1 , len(col_labels)))
        
        pol_index_mapping_dict = dict(zip(col_labels,range(len(col_labels))))
        
        for t,x in time_series_unique.values:
            time_series_CG_matrix[t,pol_index_mapping_dict[x]] = 1
            
        self.time_series_CG_matrix = pd.DataFrame(time_series_CG_matrix , columns=col_labels , dtype=int)

    def construct(self):
        """Construct causal avalanches of conflict events. These are not time
        ordered, but simply a list of the indices of all conflict events that have
        happened.
        """

        ava = []  # indices of conflict events grouped into avalanches
        event_t = []  # time index for each conflict event
        remaining_ix = set(self.time_series.index)  # events to consider
        to_check = set()
        checked = []
        tx_group = self.time_series.groupby(['t','x'])

        while remaining_ix:
            ix = remaining_ix.pop()
            t, x = self.time_series.loc[ix]
            to_check.add(ix)
            ava.append([ix])
            event_t.append([t])

            # add all the events sharing the starting time and spatial bin
            for i in tx_group.groups[(t,x)]:
                try:
                    remaining_ix.remove(i)
                    ava[-1].append(i)
                    event_t[-1].append(t)
                    checked.append(i)
                except:
                    pass
            
            # iterate thru potential sequential/preceding events
            while to_check:
                checked.append(to_check.pop())
                start_ix = checked[-1]
                t, x = self.time_series.loc[checked[-1]]

                # add successors which must be at the next time step
                for n in self.causal_graph.neighbors(x):
                    if (t+1,n) in tx_group.groups.keys():
                        # remove events such that they are not added to another avalanche
                        added = False
                        for i in tx_group.groups[(t+1,n)]:
                            try:
                                remaining_ix.remove(i)
                                # add them to the current avalanche
                                ava[-1].append(i)
                                event_t[-1].append(t)
                                checked.append(i)
                                added = True
                            except:
                                pass
                        if added:
                            # make sure successor events will be checked themselves for neighbors
                            # and only need to follow up on one in the group of successors
                            checked.pop(-1)
                            if not ava[-1][-1] in checked:
                                to_check.add(ava[-1][-1])

                # add predecessors which must be at the previous time step
                for n in self.causal_graph.predecessors(x):
                    if (t-1,n) in tx_group.groups.keys():
                        # remove events from being added to another avalanches
                        added = False
                        for i in tx_group.groups[(t-1,n)]:
                            try:
                                remaining_ix.remove(i)
                                # add them to the current avalanche
                                ava[-1].append(i)
                                event_t[-1].append(t)
                                checked.append(i)
                                added = True
                            except:
                                pass
                        if added:
                            checked.pop(-1)
                            if not ava[-1][-1] in checked:
                                to_check.add(ava[-1][-1])

        # conflict avalanches, index is index of conflict event in conflict events DataFrame
        self.avalanches = ava    ##### The event number here are the default index in ACLED dataset
        self.event_t = event_t

    def avalanche_events(self, ix):
        """Time ordered list of events in avalanche.

        Parameters
        ----------
        ix : int
            Avalanche index.

        Returns
        -------
        list of twoples
            Each twople is (t, indices of all events at time t).
        """

        a = zip(self.event_t[ix], self.avalanches[ix])
        a = sorted(a, key=lambda i:i[0])
        a_by_t = {}
        for t, i in a:
            if t in a_by_t.keys():
                a_by_t[t].append(i)
            else:
                a_by_t[t] = [i]
        return a_by_t
#end Avalanche



def pair_te(t1, t2, tmx):
    """Is t1 explained by t2 or just by itself?"""

    X = np.zeros((tmx+1, 3), dtype=int)
    X[t1-1,0] = 1 # x_{t+1}
    X[t1,1] = 1   # x_t
    X[t2,2] = 1   # y_t
    X = X[:-1]
    X = np.vstack((X, bin_states(3)))
    
    # probability distribution over all possible configurations
    # first col indicates how t1 lines up with itself over a delayed time interval
    # second col indicates dependency on t2
    pmat = np.unique(X, axis=0, return_counts=True)[1] - 1  # remove padded states
    pmat = np.vstack((pmat[::2], pmat[1::2])).T / pmat.sum()
    
    eps = np.nextafter(0, 1)
    p_terms = np.array([[pmat[0,0], (pmat[0,0]+pmat[2,0]) * pmat[0].sum()/(eps+pmat[0].sum()+pmat[2].sum())],
                        [pmat[0,1], (pmat[0,1]+pmat[2,1]) * pmat[0].sum()/(eps+pmat[0].sum()+pmat[2].sum())],
                        [pmat[1,0], (pmat[1,0]+pmat[3,0]) * pmat[1].sum()/(eps+pmat[1].sum()+pmat[3].sum())],
                        [pmat[1,1], (pmat[1,1]+pmat[3,1]) * pmat[1].sum()/(eps+pmat[1].sum()+pmat[3].sum())],
                        [pmat[2,0], (pmat[0,0]+pmat[2,0]) * pmat[2].sum()/(eps+pmat[0].sum()+pmat[2].sum())],
                        [pmat[2,1], (pmat[0,1]+pmat[2,1]) * pmat[2].sum()/(eps+pmat[0].sum()+pmat[2].sum())],
                        [pmat[3,0], (pmat[1,0]+pmat[3,0]) * pmat[3].sum()/(eps+pmat[1].sum()+pmat[3].sum())],
                        [pmat[3,1], (pmat[1,1]+pmat[3,1]) * pmat[3].sum()/(eps+pmat[1].sum()+pmat[3].sum())]])

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
        Assuming only unique and ordered values.
    tmx : int

    Returns
    -------
    float
    """
    
    (p11, p01, p10, p00), (p1_past, p1_fut) = _self_probabilities(t, tmx)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        te = np.nansum([p11 * (np.log(p11) - np.log(p1_past) - np.log(p1_fut)),
                        p10 * (np.log(p10) - np.log(p1_past) - np.log(1-p1_fut)),
                        p01 * (np.log(p01) - np.log(1-p1_past) - np.log(p1_fut)),
                        p00 * (np.log(p00) - np.log(1-p1_past) - np.log(1-p1_fut))])
    return te

def _self_probabilities(t, tmx):
    """Helper function for self_edges()."""

    # the complement of event times t
    t_comp = np.delete(np.arange(tmx+1), t)
    
    # use intersections to count possible outcomes
    # the past is the first var, the future is the second var
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
    
    if t[0]==0:
        p1_fut = (t.size-1) / tmx
    else:
        p1_fut = t.size / tmx

    if t[-1]==tmx:
        p1_past = (t.size-1) / tmx
    else:
        p1_past = t.size / tmx

    return (p11, p01, p10, p00), (p1_past, p1_fut)

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
    
    if(conflict_type == "battles"):
        df = ACLED2020.battles_df(to_lower=True)
    elif(conflict_type == "RP"):
        df = ACLED2020.riots_and_protests_df(to_lower=True)
    elif(conflict_type == "VAC"):
        df = ACLED2020.vac_df(to_lower=True)

    conflict_ev = gpd.GeoDataFrame(df[['event_date','longitude','latitude']],
                                   geometry=gpd.points_from_xy(df.longitude, df.latitude),
                                   crs=polygons.crs)
    conflict_ev['t'] = (conflict_ev['event_date']-conflict_ev['event_date'].min()) // np.timedelta64(dt,'D')
    
    conflict_ev = gpd.sjoin(conflict_ev, polygons, how='left', op='within')

    # in rare instances, a conflict event may belong to two polygons, in such a case choose the first one
    conflict_ev = conflict_ev[~conflict_ev.index.duplicated(keep='first')]

    conflict_ev.rename(columns={'index_right':'x'}, inplace=True)

    # no need for polygon neighbors column or raw index
    return conflict_ev.drop(['neighbors','index'], axis=1)

