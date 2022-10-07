from .construct import *



def test_pair_te():
    """From a calculation in pyinform."""

    xs = [0,1,1,1,1,0,0,0,0]
    ys = [0,0,1,1,1,1,0,0,0]
    assert np.isclose(pair_te(np.where(ys)[0], np.where(xs)[0], len(xs)-1)/np.log(2), .811278124)
    
