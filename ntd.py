# ====================================================================================== #
# Module for simulations on and of Nice Trees of dimension D (NTD).
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
import multiprocess as mp
from misc.stats import PowerLaw
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
        self.growingBranches = []
        self.deadBranches = []
        self.radius = []

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
                    rand_factor_fun=None,
                    reset=True):
        """Grow NTD but randomly extend branches by some factor. This is just like the
        method grow() except for the randomness in the lengths of children branches.
        
        Parameters
        ----------
        n_steps : int
        record_every : int, 10
            Record max radius every number of steps.
        reset : bool, True
            
        Returns
        -------
        ndarray
        """
        
        b, r = self.b, self.r
        rand_factor_fun = rand_factor_fun or (lambda:self.rng.uniform(1/b,b))
        counter = 1
        if reset or len(self.growingBranches)==0:
            growingBranches = [Branch('%d'%i, b) for i in range(r)]
            deadBranches = []
            radius = []
        else:
            growingBranches = self.growingBranches
            deadBranches = self.deadBranches
            radius = self.radius

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
        
        self.growingBranches = growingBranches
        self.deadBranches = deadBranches
        self.radius = radius
        return np.array(radius)
    
    def df(self):
        """Fractal dimension.
        """
        
        return 1 + np.log(self.r)/np.log(self.b)
        
    def plot(self, fig=None, ax=None,
             angle_index_offset=-1,
             unit=1,
             origin_line_width=10,
             lw_decay_exponent=.8):
        """
        Parameters
        ----------
        fig : plt.Figure, None
        ax : plt.Axes, None
        angle_index_offset : int, -1
        unit : float, 1
            Length of each piece of LineCollection used to vary thickness. This can be
            helpful if pieces are missing near the origin.
        origin_line_width : float, 10
        lw_decay_exponent : float, .8
        
        Returns
        -------
        plt.Figure
        """
        
        if fig is None and ax is None:
            fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})
        elif fig and ax is None:
            ax = fig.add_subplot(111, aspect='equal')
    
        branches = self.growingBranches + self.deadBranches
        branchids = [br.label for br in branches]

        branchidsbygen = []  # branch ids by generation
        for i in range(1, max([len(el) for el in branchids])+1):
            branchidsbygen.append([b for b in branchids if len(b)==i])

        plotAngle = {}
        rt = []  # position labeled as radius and angle pairs
        
        angle = 2 * np.pi / self.r
        for i in range(self.r):
            rt.append(((0, 0),
                      (branches[branchids.index(branchidsbygen[0][i])].pos, (i+angle_index_offset)*angle)))

        gencount = 1
        while gencount<(len(branchidsbygen)):
            radius = self.b**gencount
            angle = 2 * np.pi / self.r**gencount

            for i,bid in enumerate(branchidsbygen[gencount-1]):
                bran = branches[branchids.index(bid)]
                if gencount==1:
                    rt0 = bran.pos, i*angle
                else:
                    # branching angle is offset from parent branch
                    rt0 = (bran.ancestralLen + bran.pos, plotAngle[bran.label[:-1]] +
                                                         angle*(int(bran.label[-1]) + angle_index_offset))
                plotAngle[bid] = rt0[1]

                # find all its children and connect to them
                for j in range(self.r):
                    if bran.label+str(j) in branchids:
                        childbran = branches[branchids.index(bran.label+str(j))]
                        #if gencount==(len(branchidsbygen)-1):
                        rt.append((rt0,(childbran.ancestralLen + childbran.pos,
                                        plotAngle[childbran.label[:-1]] +
                                        angle/self.r*(int(childbran.label[-1]) + angle_index_offset))))
            gencount += 1

        xy = rt2xy(rt)

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
            lw = lw**lw_decay_exponent / origin_line_width**lw_decay_exponent * origin_line_width
            lineColl.append(LineCollection(segments, linewidths=lw))
        
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

    def plot3d(self, fig=None, ax=None,
               angle_index_offset=-1,
               unit=1,
               origin_line_width=10,
               lw_decay_exponent=.8,
               z=0,
               dx=0,
               dy=0,
               cmap=None):
        """
        Parameters
        ----------
        fig : plt.Figure, None
        ax : plt.Axes, None
        angle_index_offset : int, -1
        unit : float, 1
            Length of each piece of LineCollection used to vary thickness. This can be
            helpful if pieces are missing near the origin.
        origin_line_width : float, 10
        lw_decay_exponent : float, .8
        z : float, offset
        dx : float, 0
        dy : float, 0
        cmap
        
        Returns
        -------
        plt.Figure
        """
        
        from mpl_toolkits.mplot3d import axes3d

        if fig is None and ax is None:
            fig, ax = plt.subplots(subplot_kw={'aspect':'equal', 'projection':'3d'})
        elif fig and ax is None:
            ax = fig.add_subplot(111, aspect='equal', projection='3d')
        if cmap is None:
            cmap = plt.cm.hot
    
        branches = self.growingBranches + self.deadBranches
        branchids = [br.label for br in branches]

        branchidsbygen = []  # branch ids by generation
        for i in range(1, max([len(el) for el in branchids])+1):
            branchidsbygen.append([b for b in branchids if len(b)==i])

        plotAngle = {}
        rt = []  # position labeled as radius and angle pairs
        
        angle = 2 * np.pi / self.r
        for i in range(self.r):
            rt.append(((0, 0),
                      (branches[branchids.index(branchidsbygen[0][i])].pos, (i+angle_index_offset)*angle)))

        gencount = 1
        while gencount<(len(branchidsbygen)):
            radius = self.b**gencount
            angle = 2 * np.pi / self.r**gencount

            for i,bid in enumerate(branchidsbygen[gencount-1]):
                bran = branches[branchids.index(bid)]
                if gencount==1:
                    rt0 = bran.pos, i*angle
                else:
                    # branching angle is offset from parent branch
                    rt0 = (bran.ancestralLen + bran.pos, plotAngle[bran.label[:-1]] +
                                                         angle*(int(bran.label[-1]) + angle_index_offset))
                plotAngle[bid] = rt0[1]

                # find all its children and connect to them
                for j in range(self.r):
                    if bran.label+str(j) in branchids:
                        childbran = branches[branchids.index(bran.label+str(j))]
                        #if gencount==(len(branchidsbygen)-1):
                        rt.append((rt0,(childbran.ancestralLen + childbran.pos,
                                        plotAngle[childbran.label[:-1]] +
                                        angle/self.r*(int(childbran.label[-1]) + angle_index_offset))))
            gencount += 1

        xy = rt2xy(rt)

        lineColl = []
        for p1, p2 in xy:
            el = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            nsegments = int(el//unit)
            # linearly interpolate btwn endpts
            x = np.linspace(p1[0], p2[0], nsegments) + dx
            y = np.linspace(p1[1], p2[1], nsegments) + dy

            # taken from stackoverflow
            pts = np.vstack((x,y,np.zeros(x.size)+z)).T.reshape(-1,1,3)
            segments = np.concatenate((pts[:-1], pts[1:]), axis=1)

            # vary line thickness
            lw = np.linspace(origin_line_width/(np.linalg.norm(p1)+1),
                             origin_line_width/(np.linalg.norm(p2)+1),
                             nsegments)
            lw = lw**lw_decay_exponent / origin_line_width**lw_decay_exponent * origin_line_width
            lineColl.append(Line3DCollection(segments,
                                             linewidths=lw,
                                             colors=cmap(lw/origin_line_width)))
        
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
    
        return fig, ax
#end NTD


class ConflictReportsTrajectory(NTD):
    def __init__(self, r, b, gamma_s, theta_s, gamma_f, theta_f,
                 alpha=2.44,
                 rng=None):
        """Growing conflict trees on NTDs. Basically, NTDs but with site dynamics to count
        fatalities and reports.

        Parameters
        ----------
        r : int
            Splitting number for each branch.
        b : int
            Exponential growth base.
        gamma_s : float
            Growth exponent for reports.
        theta_s : float
        gamma_f : float
            Growth exponent for fatalities.
        theta_f : float
        alpha : float, 2.44
            Exponent for distribution of virulence.
        rng : np.random.RandomState, None
        """
        
        assert r>1 and b>1
        self.r = r
        self.b = b
        self.df = 1+np.log(r)/np.log(b)
        self.thetas = theta_s
        self.thetaf = theta_f
        self.gammas = gamma_s
        self.gammaf = gamma_f
        self.rng = rng or np.random

        self.alpha = alpha

    def grow(self,
             threshold_s,
             v_s=1,
             v_f=1,
             v_s0=1,
             s0=1/15,
             record_every=10,
             mx_rand_coeff=None,
             mx_counter=np.inf):
        """Grow NTD til the rate of events added from new sites joining at the periphery
        falls below a fixed threshold. This is different from dynamics determined by
        endemic conflict event generation at each site.

        [total rate] = virulence * ( [endemic site rate] + [mean new site rate] )
        
        Parameters
        ----------
        threshold_s : float
            Rate threshold below which conflict avalanche stops.
        v_s : float or tuple, 1
            If tuple, determines parameters for random sampling.
        v_f : float or tuple, 1
        v_s0 : float, 1
            Lower limit for distribution of v_s.
        s0 : float, 1/15
            Default value from calculation of averaged s0 (Conflict III, pg. 90).
        record_every : int, 10
            Record max radius every number of steps.
        mx_rand_coeff : float, self.b
            Coefficient for determining magnitude of random fluctuations in lengths of
            branches.
        mx_counter : int, np.inf
            Max number of iterations to allow.
            
        Returns
        -------
        ndarray
            Max radius (as measured from origin) per time step.
        ndarray
            Cumulative number of reports per time step.
        ndarray
            Cumulative number of fatalities per time step.
        """
        
        b, r, ths, thf, g_s, g_f = self.b, self.r, self.thetas, self.thetaf, self.gammas, self.gammaf
        mx_rand_coeff = mx_rand_coeff or b
        assert mx_counter>2

        growingBranches = [Branch('%d'%i, b) for i in range(r)]
        deadBranches = []
        radius = [0]
        activeSites = [Site(1, v_s, g_s, ths, v_f, g_f, thf)]
        deadSites = [Site(0, v_s, g_s, ths, v_f, g_f, thf)]  # seed site
        counter = 2  # time counter
        
        # grow randomly branching tree
        while ((v_s * counter**(1 - 2 / self.df)) > threshold_s) and counter<=mx_counter:
            # randomly select a branch to extend
            randix = self.rng.randint(len(growingBranches))
            gb = growingBranches[randix]

            # try to extend selected branch
            # if branching point reached, then spawn children, remove dead branch, and try again
            if not gb.grow():
                # create r children of this branch
                for j in range(r):
                    # new branches have random length
                    nb = Branch('%s%d'%(gb.label,j),
                                int(b**(len(gb.label)+1) * self.rng.uniform(1/mx_rand_coeff, mx_rand_coeff)),
                                ancestral_length=gb.len+gb.ancestralLen)
                    growingBranches.append(nb)
                deadBranches.append(growingBranches.pop(randix))

            # else successfully added new conflict site and start generating events
            # time only ticks in this portion of the loop
            else:
                # generate new site that starts at t=counter
                activeSites.append( Site(counter, v_s, g_s, ths, v_f, g_f, thf) )

                if (counter%record_every)==0:
                    radius.append(max([gb.pos+gb.ancestralLen for gb in growingBranches]))
                counter += 1
        
        deadSites += activeSites
        radius = np.array(radius)
        
        # calculate size and fat trajectories in discrete units
        cumS = np.zeros(counter, dtype=int)  # by defn, avalanche must start with at least 1 event
        cumF = np.zeros(counter, dtype=int)
        for i, site in enumerate(deadSites):
            # the seed site comes with at least one event by definition
            if i==0:
                startingReportCount = 1
                cumS[site.t0:] += (site.cum(np.arange(site.t0, counter), t0_offset=1).astype(int) +
                                   startingReportCount)
                cumF[site.t0:] += site.cum_f(np.arange(site.t0, counter), t0_offset=1).astype(int)
            else:
                # this check only applies with v_s is randomly sampled
                assert site.v_s>=v_s0
                # starting report count depends on virulence v_s and s0 (could be 0)
                startingReportCount = self.rng.poisson(v_s / v_s0 * s0)  
                cumS[site.t0:] += (site.cum(np.arange(site.t0, counter), t0_offset=1).astype(int) +
                                   startingReportCount)
                cumF[site.t0:] += site.cum_f(np.arange(site.t0, counter), t0_offset=1).astype(int)
        
        # save status of sim
        self.growingBranches = growingBranches
        self.deadBranches = deadBranches
        self.radius = radius
        self.deadSites = deadSites

        return radius, cumS, cumF
        
    def sample(self, n_samples, c0,
               v_s0=0.01,
               v_s1=10_000,
               record_every=1,
               iprint=True,
               n_cpus=None,
               **grow_kw):
        """Generate many example trajectories.

        Randomly sampling from virulence distribution given
            V_s ~ T,
            V_f ~ T^{3/2},
            C_s = C_0 * T^{2 - 2/z}
        
        Parameters
        ----------
        n_samples : int
        c0 : float
            Threshold coefficient for when avalanche ends. This is multiplied by a scaling
            function of total duration to get cutoff for any given avalanche simulation.
        v_s0 : float, 0.01
        v_s1 : float, 10_000
        record_every : int, 1
        iprint : bool, True
        n_cpus : int, None
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
        assert c0>0
        assert 0<v_s0<v_s1
        assert record_every>=1
        n_cpus = n_cpus or mp.cpu_count()-1

        def loop_wrapper(args, self=self):
            i, v_s = args
            self.rng = np.random.RandomState()
            v_f = (v_s / v_s0)**1.5  # this relationship is from measuring d_F/z - delta_f/zeta
            
            # must be in the limit of single events per site in periphery
            # cutoff scales with expected duration
            rt, st, ft = self.grow(c0 * (v_s/v_s0)**(2 - 2/self.df), v_s, v_f,
                                   v_s0=v_s0,
                                   record_every=record_every)
            n = len(self.deadSites)
            # total number of events per site x
            sx = np.array([site.cum(st.size+1) for site in self.deadSites])
            
            # print mod10 counter
            if i>0 and ((i+1)%10)==0 and iprint: print('Done with %d'%(i+1))

            return rt, st, ft, n, sx
        
        # sample from virulence
        pls = PowerLaw(self.alpha, lower_bound=v_s0, upper_bound=v_s1)
        v_s = pls.rvs(size=n_samples)

        # run parallelized sampling procedure
        try:
            pool = mp.Pool(n_cpus)
            r, s, f, n, sx = list(zip(*pool.map(loop_wrapper, enumerate(v_s))))
        finally:
            pool.close()
            
        return r, s, f, v_s, np.array(n), sx
#end ConflictReportsTrajectory

CRT = ConflictReportsTrajectory


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

# ============== #
# Helper classes #
# ============== #
class Site():
    """Conflict site with accumulation dynamics for both reports and fatalities."""
    def __init__(self,
                 t0,
                 v_s,
                 gamma_s,
                 theta_s,
                 v_f,
                 gamma_f,
                 theta_f,
                 rng=None):
        """
        Parameters
        ----------
        t0 : int
            Time at which conflict site was born.
        v_s : float or tuple
            Quenched disorder.
        gamma_s : float
            Rate exponent.
        theta_s : float
            Suppression exponent.
        v_f : float or tuple
        gamma_f : float
        theta_f : float
        rng : np.random.RandomState, None
        """

        assert theta_s>0 and gamma_s<0
        assert theta_f>0 and gamma_f<0

        self.t0 = t0
        self.gamma_s = gamma_s
        self.theta_s = theta_s
        self.gamma_f = gamma_f
        self.theta_f = theta_f
        self.rng = rng or np.random

        if hasattr(v_s,'__len__'):
            self.v_s = self.rng.exponential(scale=v_s[0])
        else:
            assert v_s>0
            self.v_s = v_s

        if hasattr(v_f,'__len__'):
            self.v_f = self.rng.exponential(scale=v_f[0])
        else:
            assert v_f>0
            self.v_f = v_f

        self.rate = self.rate_s
        self.cum = self.cum_s

    def rate_s(self, t, t0_offset=1):
        """Instantaneous rate of conflict events.
        """
        
        self.check_t(t)
        return self.v_s * (t-self.t0+t0_offset)**self.gamma_s * (self.t0+t0_offset)**-self.theta_s

    def cum_s(self, t, t0_offset=1):
        """Cumulative conflict events.
        """

        self.check_t(t)
        return self.v_s * (t-self.t0+t0_offset)**(1+self.gamma_s) * (self.t0+t0_offset)**-self.theta_s

    def rate_f(self, t, t0_offset=1):
        """Instantaneous rate of conflict events.
        """

        self.check_t(t)
        return self.v_f * (t-self.t0+t0_offset)**self.gamma_f * (self.t0+t0_offset)**-self.theta_f

    def cum_f(self, t, t0_offset=1):
        """Cumulative conflict events.
        """

        self.check_t(t)
        return self.v_f * (t-self.t0+t0_offset)**(1+self.gamma_f) * (self.t0+t0_offset)**-self.theta_f
    
    def check_t(self, t):
        if hasattr(t, '__len__'):
            assert (t>=self.t0).all()
        else:
            assert t>=self.t0, (t, self.t0)
