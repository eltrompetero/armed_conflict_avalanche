from pyutils.acled_utils import PoissonDiscSphere
from numpy import pi
import numpy as np


def test_PoissonDiscSphere():
    poissd=PoissonDiscSphere(pi/50,
                             fast_sample_size=5,
                             width_bds=(0,.5),
                             height_bds=(0,.5))
    poissd.sample()

    # make sure that closest neighbor is the closest one in the entire sample
    pt=np.array([.2,.3])
    nearestix=poissd.get_closest_neighbor(pt)
    d=poissd.dist(pt,poissd.samples)
    assert nearestix==np.argmin(d)
