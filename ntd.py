# ====================================================================================== #
# Module for simulations on and of Nice Trees of dimension D (NTD).
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np


class Branch():
    """A single branch.
    """
    def __init__(self, label, length,
                 ancestral_length=0):
        self.label = label
        self.len = length
        self.pos = 0
        self.ancestralLen = ancestral_length
    
    def grow(self, step_size=1):
        """Move out away from the starting point of the branch.
        
        Parameters
        ----------
        step_size : int, 1
        
        Returns
        -------
        bool
            True if branch has not exceeded length.
        """
        
        self.pos += step_size
        if self.pos>self.len:
            self.pos = self.len
            return False
        return True
#end Branch


class NTD():
    def __init__(self, r, b, rng=None):
        """
        r : int,
            Splitting number.
        b : int
            Exponential growth base.
        rng : np.random.RandomState
        """
        
        assert r>1 and b>1
        self.r = r
        self.b = b
        self.rng = rng or np.random.RandomState()

    def grow(self, n_steps,
             record_every=10):
        """Grow NTD.
        
        Parameters
        ----------
        n_steps : int
        record_every : int, 10
            Record max radius every number of steps.
            
        Returns
        -------
        ndarray
        """
        
        b, r = self.b, self.r
        counter = 1
        growingBranches = [Branch('%d'%i, b) for i in range(r)]
        deadBranches = []
        radius = []

        while counter<n_steps:
            # iterate through the current set of branches that are growing without
            # considering new branches that are added in this loop
            n = len(growingBranches)
            i = 0  # counter for size of current generation
            while i<n:
                gb = growingBranches[i]
                # increment branch but note whether or not it has reached a branch point
                if not gb.grow():
                    # create all children of this branch
                    newBranches = []
                    for j in range(r):
                        # new branches have random length
                        nb = Branch('%s%d'%(gb.label,j),
                                    b**(len(gb.label)+1),
                                    ancestral_length=gb.len+gb.ancestralLen)
                        newBranches.append(nb)
                    growingBranches += newBranches
                    deadBranches.append(growingBranches.pop(i))
                    n -= 1
                    i -= 1
                i += 1

                if (counter%record_every)==0:
                    el = [gb.pos+gb.ancestralLen for gb in growingBranches]
                    radius.append(max(el))
                counter += 1
        radius = np.array(radius)
        
        self.growingBranches = growingBranches
        self.deadBranches = deadBranches
        self.radius = radius
        return radius

    def grow_random(self, n_steps,
                    record_every=10,
                    rand_factor_fun=None):
        """Grow NTD but randomly extend branches by some factor. This is just like the
        method grow() except for the randomness in the lengths of children branches.
        
        Parameters
        ----------
        n_steps : int
        record_every : int, 10
            Record max radius every number of steps.
            
        Returns
        -------
        ndarray
        """
        
        b, r = self.b, self.r
        rand_factor_fun = rand_factor_fun or (lambda:self.rng.uniform(1/b,b))
        counter = 1
        growingBranches = [Branch('%d'%i, b) for i in range(r)]
        deadBranches = []
        radius = []

        while counter<n_steps:
            # iterate through the current set of branches that are growing without
            # considering new branches that are added in this loop
            n = len(growingBranches)
            i = 0  # counter for size of current generation
            while i<n:
                gb = growingBranches[i]
                # increment branch but note whether or not it has reached a branch point
                if not gb.grow():
                    # create all children of this branch
                    newBranches = []
                    for j in range(r):
                        # new branches have random length
                        nb = Branch('%s%d'%(gb.label,j),
                                    int(b**(len(gb.label)+1)*rand_factor_fun()),
                                    ancestral_length=gb.len+gb.ancestralLen)
                        newBranches.append(nb)
                    growingBranches += newBranches
                    deadBranches.append(growingBranches.pop(i))
                    n -= 1
                    i -= 1
                i += 1

                if (counter%record_every)==0:
                    el = [gb.pos+gb.ancestralLen for gb in growingBranches]
                    radius.append(max(el))
                counter += 1
        radius = np.array(radius)
        
        self.growingBranches = growingBranches
        self.deadBranches = deadBranches
        self.radius = radius
        return radius
    
    def df(self):
        """Fractal dimension.
        """
        
        return 1 + np.log(self.r)/np.log(self.b)
#end NTD
