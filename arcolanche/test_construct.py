from .construct import *



def test_pair_te():
    """From a calculation in pyinform."""

    xs = [0,1,1,1,1,0,0,0,0]
    ys = [0,0,1,1,1,1,0,0,0]
    assert np.isclose(pair_te(np.where(ys)[0], np.where(xs)[0], len(xs)-1)/np.log(2), .811278124)
    
def test_avalanches():
    ava = Avalanche(64, 80, sig_threshold=95, gridix=1)
    check_for_split_avalanches(ava)

def check_for_split_avalanches(ava):
    """
    Parameters
    ----------
    ava : arcolanche.contruct.Avalanche
    """
    
    avalanches = sorted(ava.avalanches, key=len)
    
    for ava_ix in range(len(avalanches)):
        # for this avalanche, get its time and spatial bin
        a = avalanches[ava_ix]

        bins = np.unique(ava.time_series.loc[a], axis=0)
        t = bins[:,0]
        x = bins[:,1]

        # check whether or not this time/space bin occured either coincident with another avalanche
        # previous or after
        for i in range(len(avalanches)):
            if ava_ix!=i:
                a = avalanches[i]

                bins = np.unique(ava.time_series.loc[a], axis=0)
                this_t = bins[:,0]
                this_x = bins[:,1]

                # check for coincidence
                for t_, x_ in zip(t, x):
                    assert ((this_t!=t_) | (this_x!=x_)).all(), f"Problem with {ava_ix}, {i}."

                # check for antecedence (if focus avalanche is child of later events)
                for t_, x_ in zip(t, x):
                    assert not ((this_t==t_-1) &
                                np.array([this_x_ in set(ava.causal_graph.predecessors(x_))
                                          for this_x_ in this_x])).any(), f"Problem with {ava_ix}, {i}."

                # check for descendance (if focus avalanche is parent of later events)
                for t_, x_ in zip(t, x):
                    assert not ((this_t==t_+1) &
                                np.array([this_x_ in set(ava.causal_graph.neighbors(x_))
                                          for this_x_ in this_x])).any(), f"Problem with {ava_ix}, {i}."
