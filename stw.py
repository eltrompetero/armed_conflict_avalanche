# ===================================================================================== #
# Self-trapping walker simulations.
# Author: Eddie Lee, edlee@santafe.edu
# ===================================================================================== #
import numpy as np
from numpy import maximum
import multiprocess as mp



class STW1D():
    """Self-trapping walker with periodic boundaries.

    In comparison with "self-avoiding walk," self-trapping refers to a process that is
    more likely to stay where it's already visited.
    """
    def __init__(self, L, alpha,
                 rng=None):
        self.L = L
        self.alpha = alpha
        self.rng = rng or np.random.RandomState()
        
    def run(self, T, recordEvery=10):
        x = self.L//2  # position of walker on lattice
        r = np.ones(self.L)  # number of repeats on sites
        
        el = np.zeros(T//recordEvery, dtype=int)
        total = np.zeros(T//recordEvery)

        counter = 0
        while counter < T:
            # account for the current location of the particle
            r[x] += 1

            # move the particle to the next location with rate proportional to r
            a = self.rng.rand()
            den = r[(x-1)%self.L]**self.alpha + r[x]**self.alpha + r[(x+1)%self.L]**self.alpha
            if a > (r[x]**self.alpha/den):
                if (a-r[x]**self.alpha/den) < (r[(x-1)%self.L]**self.alpha/den):
                    x -= 1
                else:
                    x += 1
                x %= self.L

            el[counter//recordEvery] = x
            total[counter//recordEvery] = r.sum() - self.L

            counter += 1

        el = maximum.accumulate(abs(el - self.L//2))
        self.r = r
        return el, total, r
    
    def sample(self, n_sample, T,
               **run_kw):
        """Generate n_sample trajectories of duration T.
        """
        
        def wrapper(i):
            self.rng = np.random.RandomState()
            return self.run(T, **run_kw)
            
        with mp.Pool(mp.cpu_count()-1) as pool:
            el, total, r = list(zip(*pool.map(wrapper, range(n_sample))))
            
        return el, total, r
    
    def sample_df(self, n_sample, T,
                  **run_kw):
        """Sample to estimate fractal dimension."""
        
        el, total = self.sample(n_sample, T)[:-1]
        el = np.array([i[-1] for i in el])
        total = np.array([t[-1] for t in total])
        
        return np.log(total.mean()) / np.log(el.mean())
#end STW1D
