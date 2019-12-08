# ====================================================================================== #
# Module for simulations on and of Nice Trees of dimension D (NTD).
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
import multiprocess as mp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


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
                    for j in range(r):
                        # new branches have random length
                        nb = Branch('%s%d'%(gb.label,j),
                                    b**(len(gb.label)+1),
                                    ancestral_length=gb.len+gb.ancestralLen)
                        nb.grow()
                        growingBranches.append(nb)
                    deadBranches.append(growingBranches.pop(i))
                    n -= 1
                    i -= 1
                if (counter%record_every)==0:
                    el = [gb.pos+gb.ancestralLen for gb in growingBranches]
                    radius.append(max(el))
                counter += 1
                i += 1

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
            randix = self.rng.randint(len(growingBranches))
            gb = growingBranches[randix]
            # increment branch but note whether or not it has reached a branch point
            if not gb.grow():
                # create all children of this branch
                for j in range(r):
                    # new branches have random length
                    nb = Branch('%s%d'%(gb.label,j),
                                int(b**(len(gb.label)+1)*rand_factor_fun()),
                                ancestral_length=gb.len+gb.ancestralLen)
                    nb.grow()
                    growingBranches.append(nb)
                deadBranches.append(growingBranches.pop(randix))
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
        
    def plot(self, fig=None, ax=None,
             angle_index_offset=-1,
             origin_line_width=10):
        """
        Parameters
        ----------
        fig : plt.Figure, None
        ax : plt.Axes, None
        angle_index_offset : int, -1
        origin_line_width : float, 10
        
        Returns
        -------
        plt.Figure
        """
        
        if fig is None and ax is None:
            fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})
        elif fig and ax is None:
            ax = fig.add_subplot(111, aspect='equal')
    
        branches = self.growingBranches + self.deadBranches
        branchids = [br.label for br in branches]  # branch ids by generation

        branchidsbygen = []
        for i in range(1, max([len(el) for el in branchids])+1):
            branchidsbygen.append([b for b in branchids if len(b)==i])

        plotAngle = {}
        rt = []  # position labeled as radius and angle pairs

        gencount = 1
        while gencount<(len(branchidsbygen)):
            radius = self.b**gencount
            angle = 2 * np.pi / self.r**gencount

            for i,bid in enumerate(branchidsbygen[gencount-1]):
                bran = branches[branchids.index(bid)]
                if gencount==1:
                    rt0 = bran.ancestralLen, i*angle
                else:
                    rt0 = (bran.ancestralLen, plotAngle[bran.label[:-1]] +
                                              angle*(int(bran.label[-1]) + angle_index_offset))
                plotAngle[bid] = rt0[1]

                # find all its children and connect to them
                for j in range(self.r):
                    if bran.label+str(j) in branchids:
                        childbran = branches[branchids.index(bran.label+str(j))]
                        rt.append((rt0,(childbran.ancestralLen,
                                        plotAngle[childbran.label[:-1]] +
                                        angle/self.r*(int(childbran.label[-1]) + angle_index_offset))))

            gencount += 1

        xy = rt2xy(rt)
        
        unit = 1
        lineColl = []
        for p1,p2 in xy:
            el = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            nsegments = int(el//unit)
            # linearly interpolate btwn endpts
            x = np.linspace(p1[0], p2[0], nsegments)
            y = np.linspace(p1[1], p2[1], nsegments)

            # taken from stackoverflow
            pts = np.vstack((x,y)).T.reshape(-1,1,2)
            segments = np.concatenate((pts[:-1], pts[1:]), axis=1)

            # vary line thickness
            lw = np.linspace(origin_line_width/(np.linalg.norm(p1)+1),
                             origin_line_width/(np.linalg.norm(p2)+1),
                             nsegments)
            lineColl.append(LineCollection(segments, linewidths=lw**.8))
        
        # plotting limits
        xy = np.vstack(xy)
        xlim = (xy[:,0].min()-1, xy[:,0].max()+1)
        ylim = (xy[:,1].min()-1, xy[:,1].max()+1)
        
        # plot
        for lc_ in lineColl:
            ax.add_collection(lc_)
        ax.set(xlim=xlim, ylim=ylim)

        ax.set(xticks=[], yticks=[])
        [s.set_visible(False) for s in ax.spines.values()]
    
        return fig
#end NTD


class ConflictReportsTrajectory(NTD):
    def __init__(self, r, b, theta, gammas, gammaf, rng=None):
        """
        Parameters
        ----------
        r : int
            Splitting number.
        b : int
            Exponential growth base.
        theta : float
        gammas : float
            Growth exponent for reports.
        gammaf : float
            Growth exponent for fatalities.
        rng : np.random.RandomState
        """
        
        assert r>1 and b>1
        self.r = r
        self.b = b
        self.theta = theta
        self.gammas = gammas
        self.gammaf = gammaf
        self.rng = rng or np.random.RandomState()

    def grow(self, n_steps,
             record_every=10,
             mx_rand_coeff=None):
        """Grow NTD while keeping track of the number of conflict reports at each site.
        
        Parameters
        ----------
        n_steps : int
        record_every : int, 10
            Record max radius every number of steps.
        max_rand_coeff : float, self.b
            
        Returns
        -------
        ndarray
            Max radius (as measured from origin) per time step.
        ndarray
            Cumulative number of reports per time step.
        ndarray
            Cumulative number of fatalities per time step.
        """
        
        cumF = np.zeros(n_steps, dtype=int)
        cumS = np.zeros(n_steps, dtype=int)
        b, r, c, gs, gf = self.b, self.r, self.theta, self.gammas, self.gammaf
        mx_rand_coeff = mx_rand_coeff or b
        counter = 0
        growingBranches = [Branch('%d'%i, b) for i in range(r)]
        deadBranches = []
        radius = []

        while counter<n_steps:
            # randomly select a branch to add onto
            randix = self.rng.randint(len(growingBranches))
            gb = growingBranches[randix]
            # try to increment branch
            # if branching point reached, then spawn children, remove dead branch, and try again
            if not gb.grow():
                # create all children of this branch
                for j in range(r):
                    # new branches have random length
                    nb = Branch('%s%d'%(gb.label,j),
                            int(b**(len(gb.label)+1) * self.rng.uniform(1/mx_rand_coeff, mx_rand_coeff)),
                            ancestral_length=gb.len+gb.ancestralLen)
                    growingBranches.append(nb)
                deadBranches.append(growingBranches.pop(randix))
            # else successfully added new conflict site and start generating events
            else: 
                # count all future events generated by this new conflict site
                cumS[counter:] += (np.arange(n_steps-counter)**(1+gs) * (counter+1)**-c).astype(int)
                # at least one event per site
                cumS[counter:] += 1
                cumF[counter:] += (np.arange(n_steps-counter)**(1+gf) * (counter+1)**-c).astype(int)
                
                if (counter%record_every)==0:
                    radius.append(max([gb.pos+gb.ancestralLen for gb in growingBranches]))
                counter += 1
        radius = np.array(radius)
        
        self.growingBranches = growingBranches
        self.deadBranches = deadBranches
        self.radius = radius
        return radius, cumS, cumF
        
    def sample(self, n_samples, durations,
               record_every=10,
               iprint=True,
               **grow_kw):
        """Generate many example trajectories.
        
        Parameters
        ----------
        n_samples : int
        durations : int or ndarray
        record_every : int, 10
        iprint : bool, True
        **grow_kw
        
        Returns
        -------
        list of ndarray
            Max radius trajectories.
        list of ndarray
            Cumulative size trajectories.
        list of ndarray
            Cumulative fatalities trajectories.
        """
        
        assert n_samples>0
        if type(durations) is int:
            assert durations>0
            durations = np.zeros(n_samples)+durations
        elif type(durations) is np.ndarray:
            assert (durations>0).all()
        else:
            raise Exception("Unrecognized type for durations.")

        def loop_wrapper(t, self=self):
            self.rng = np.random.RandomState()
            r, s, f = self.grow(t, record_every=record_every, **grow_kw)
            if iprint:
                print("Done with avalanche of duration %d."%t)
            return r, s, f

        try:
            pool = mp.Pool(mp.cpu_count()-1)
            r, s, f = list(zip(*pool.map(loop_wrapper, durations)))
        finally:
            pool.close()
            
        return r, s, f
#end ConflictReportsTrajectory


# ================ #
# Helper functions #
# ================ #
def rt2xy(rt):
    rt = np.vstack(rt).reshape(len(rt), 4)
    xy = np.zeros_like(rt)
    xy[:,0] = rt[:,0]*np.cos(rt[:,1])
    xy[:,1] = rt[:,0]*np.sin(rt[:,1])
    xy[:,2] = rt[:,2]*np.cos(rt[:,3])
    xy[:,3] = rt[:,2]*np.sin(rt[:,3])
    xy = [((row[0],row[1]),(row[2],row[3])) for row in xy]

    return xy
