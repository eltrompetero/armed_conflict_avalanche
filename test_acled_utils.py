# ====================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from pyutils.acled_utils import PoissonDiscSphere
from numpy import pi
from .acled_utils import *
from data_sets.acled import ACLED2020



def test_grid_split2cluster_ix():
    np.random.seed(0)
    df = ACLED2020.battles_df().iloc[:1000]
    
    # generate random conflict clusters
    gridsplitix = np.random.permutation(df.index.values)
    gridsplitix = [gridsplitix[i*100:(i+1)*100] for i in range(10)]

    df['cluster'] = grid_split2cluster_ix(gridsplitix, df.index)
    fatalitiesAuto = df.groupby('cluster')['fatalities'].sum().values
    fatalitiesManual = np.array([df.fatalities.loc[ix].sum() for ix in gridsplitix])
    
    assert fatalitiesAuto.sum()==df.fatalities.sum()
    assert fatalitiesManual.sum()==df.fatalities.sum()
    assert np.array_equal(np.sort(fatalitiesAuto), np.sort(fatalitiesManual))

def test_PoissonDiscSphere():
    poissd = PoissonDiscSphere(pi/50,
                               fast_sample_size=5,
                               width_bds=(0,.5),
                               height_bds=(0,.5))
    poissd.sample()

    # make sure that closest neighbor is the closest one in the entire sample
    pt = np.array([.2,.3])
    nearestix = poissd.get_closest_neighbor(pt)
    d = poissd.dist(pt,poissd.samples)
    assert nearestix==np.argmin(d)
