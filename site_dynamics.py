# ====================================================================================== #
# Module for conflict avalanche site dynamics.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from pyutils.acled_utils import *
import pyutils.pipeline as pipe
from data_sets.acled import *
from misc.globe import NoVoronoiTilesRemaining
from data_sets.acled.coarse_grain import pixelate_voronoi
import multiprocess as mp
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize, brute
import matplotlib.pyplot as plt

# Define global variables
MIN_NO_GEO_PTS = 3
MIN_DAYS = 4



class ThetaFitter():
    """Class for fitting the peripheral suppression exponent theta.
    
    In order to use this class, must already know what gammaplus1 is beforehand. Using
    this value, one should solve for the best fit theta using solve_theta(). If a value
    for the vertical offset, c, is already known, as might be the case when calculating it
    explicitly from data, then this can be specified.
    """
    def __init__(self, trajectories, cluster_T, g,
                 gammaplus1=0.22,
                 weighted=True):
        """
        Parameters
        ----------
        trajectories : list of ndarray
            Each ndarray is a time series of number of events by day at the conflict site
            from t0 til T.
        cluster_T : np.ndarray
            Duration of the entire cluster to which each site belongs.
        g : np.ndarray
            Normalized start time (t0/T) as in notes.
        gammaplus : float, 0.22
        weighted : bool, True
            If True, standardized errors are minimized.
        """
        
        self.trajectories = trajectories
        self.clusterT = cluster_T
        self.g = g
        self.gammaplus1 = gammaplus1
        self.weighted = weighted
        
        # default number of points to fit
        self.bins = np.linspace(0, 1+1e-10, 10)  # cannot fit to singularities
        self.binsix = np.digitize(g, self.bins)
        self.binsx = (self.bins[:-1] + self.bins[1:]) / 2
    
    def avg_profile(self, theta=None):
        """
        Parameters
        ----------
        theta : float, None

        Returns
        -------
        ndarray
        ndarray
        """

        theta = theta or self.theta

        avgdur = np.zeros(self.binsix.max()+1)
        stddur = np.zeros(self.binsix.max()+1)
        d = self.gammaplus1 - theta

        for i in np.unique(self.binsix):
            avgdur[i] = np.mean([self.trajectories[ix].sum() / self.clusterT[ix]**d
                                 for ix in np.where(self.binsix==i)[0]])
            stddur[i] = self.std_of_mean([self.trajectories[ix].sum() / self.clusterT[ix]**d
                                          for ix in np.where(self.binsix==i)[0]])
        avgdur[avgdur==0] = np.nan

        return avgdur, stddur

    def cost_given_theta(self, theta,
                         c=None,
                         full_output=False):
        """Calculate best fit for a given value of theta to find optimal theta
        that fits scaling collapse.

        See solve_theta().
        
        Parameters
        ----------
        theta : float
            Exponent to collapse a * (1-f+b)^{1+gamma} * (b+f)^-theta + c
        weighted : bool, True
            Normalized error bars by uncertainty (more uncertainty means weaker
            constraint.
        c : float, None
            If this is specified, then it is set to a fixed value for the optimization
            problem.
        full_output : bool, False
            If True, full result from scipy.optimize.minimize will be returned instead of
            just the function value.
            
        Returns
        -------
        float
            Estimate of theta.
        """
        
        avgdur, stddur = self.avg_profile(theta)

        cost = self._define_cost_for_a_b_c(theta, avgdur, stddur, fix_c=c)
        if c:
            soln = minimize(cost, [50, -50])
        else:
            soln = minimize(cost, [50, -5, 100])

        if full_output:
            return soln
        return soln['fun']

    def _define_cost_for_a_b_c(self, theta, avgdur, stddur, fix_c=None):
        if fix_c:
            if fix_c<0 : raise Exception

            def cost(args,
                     theta=theta,
                     gammaplus1=self.gammaplus1,
                     binsx=self.binsx,
                     avgdur=avgdur,
                     stddur=stddur,
                     weighted=self.weighted,
                     c=fix_c):
                """Cost for finding coefficient a for scaling form of density of events 
                by scaled time.
                """
                
                a, logb = args
                if a<=0 or logb>10 : return 1e30

                # weighted
                if weighted:
                    return  (np.linalg.norm( ((1 - binsx + np.exp(logb))**gammaplus1 *
                                             (np.exp(logb) + binsx)**-theta * a + c - avgdur[1:])/stddur[1:] ) +
                             np.exp(logb))  # linear cost on large b
                # unweighted
                return np.linalg.norm( (1 - binsx + np.exp(logb))**gammaplus1 *
                                        (np.exp(logb) + binsx)**-theta * a + c - avgdur[1:] ) + np.exp(logb)
        else:
            def cost(args,
                     theta=theta,
                     gammaplus1=self.gammaplus1,
                     binsx=self.binsx,
                     avgdur=avgdur,
                     stddur=stddur,
                     weighted=self.weighted):
                """Cost for finding coefficient a for scaling form of density of events 
                by scaled time.
                """
                
                a, logb, c = args
                if a<=0 or logb>10 or c<0: return 1e30

                # weighted
                if weighted:
                    return  (np.linalg.norm( ((1 - binsx + np.exp(logb))**gammaplus1 *
                                             (np.exp(logb) + binsx)**-theta * a + c - avgdur[1:])/stddur[1:] ) +
                             np.exp(logb))  # linear cost on large b
                # unweighted
                return np.linalg.norm( (1 - binsx + np.exp(logb))**gammaplus1 *
                                        (np.exp(logb) + binsx)**-theta * a + c - avgdur[1:] ) + np.exp(logb)
        return cost
    
    def solve_theta(self, c=None):
        """Find optimal value of theta using grid search algorithm.

        Returns
        -------
        float
            Theta
        tuple
            a, logb, c
        """

        self.theta = brute(self.cost_given_theta, ([-.4,-.8],), Ns=5)[0]
        if c:
            self.a, self.logb = self.cost_given_theta(self.theta, c=c, full_output=True)['x']
            self.c = c
        else:
            self.a, self.logb, self.c = self.cost_given_theta(self.theta, full_output=True)['x']
        return self.theta, (self.a, self.logb, self.c)

    def plot(self, errs,
             theta_lb=None,
             theta_ub=None):
        """Plot fit to data alongside given data.

        Parameters
        ----------
        stddur : ndarray
        errs : ndarray
        theta_lb : float, None
            This is plotted if both theta_lb and theta_ub are specified.
        theta_ub : float, None

        Returns
        -------
        mpl.Figure
        """
        
        avgdur, stddur = self.avg_profile(self.theta)

        # setup
        b = np.exp(self.logb)
        x = np.linspace(.02, 1, 1000)
        h = []
        fig, ax = plt.subplots()
        
        # plot data and errorbars
        h.append( ax.errorbar(self.binsx, avgdur[1:], errs[:,1:], fmt='o') )
        
        # plot model
        h.append(ax.plot(x, (1 - x + b)**self.gammaplus1 * (b + x)**-self.theta * self.a + self.c, 'k-')[0])
        if not theta_lb is None and not theta_ub is None:
            h.append(ax.plot(x, (1 - x + b)**gammaplus1 * (b + x)**-theta_lb * a + self.c, 'k-.')[0])
            ax.plot(x, (1 - x + b)**gammaplus1 * (b + x)**theta_ub * a + self.c, 'k-.')
        h.append(ax.plot(x, (1 - x + b)**self.gammaplus1 * self.a + self.c, 'k--')[0])
        
        # plot properties
        ax.set(xlabel=r'relative starting time $g=t_0/T$',
               ylabel=r'$\widebar{s_{x_i}(t_0/T})/T^{1+\gamma_s-\theta_s}$',
               xlim=(-.05,1.05),
               yscale='log')
        if not theta_lb is None and not theta_ub is None:
            ax.legend(h, 
                      ('Data',
                       r'$\theta=%1.2f$'%self.theta,
                       r'$\theta^+$, $\theta^-$',
                       r'$\theta=0$',
                       r'$\langle s(t_0)\rangle$'),
                      fontsize='small', handlelength=1, loc=3, ncol=2, columnspacing=.55)
        else:
            ax.legend(h, 
                      ('Data',
                       r'$\theta=%1.2f$'%self.theta,
                       r'$\theta=0$',
                       r'$\langle s(t_0)\rangle$'),
                      fontsize='small', handlelength=1, loc=3, ncol=2, columnspacing=.55)

        return fig

    def pipeline_theta_estimate_bootstrap(self, n_samples):
        raise NotImplementedError
        theta = np.zeros(n_samples)
        
        def one_loop_wrapper(randcounter):
            np.random.seed()
            
            randix = np.random.randint(fractionalt0.size, size=fractionalt0.size)
            fractionalt0_ = fractionalt0[randix]
            num = [traj[res][ix] for ix in randix]
            den = [wholeClusterDur[ix] for ix in randix]
            
            binsix = np.digitize(fractionalt0_, bins)

            def cost_given_theta(theta, weighted=True):
                avgdur = np.zeros(binsix.max()+1)
                stddur = np.zeros(binsix.max()+1)
                exponent = gammaplus1 - theta  # 1+gamma-theta from Conflict II pg. 121

                for i in unique(binsix):
                    avgdur[i] = np.mean([num[ix].sum()/den[ix]**exponent
                                         for ix in where(binsix==i)[0]])
                    stddur[i] = self.std_of_mean([num[ix].sum()/den[ix]**exponent
                                                  for ix in where(binsix==i)[0]])
                avgdur[avgdur==0] = np.nan

                cost = self.define_cost_for_a_b(theta, avgdur, stddur)
                soln = minimize(cost, [50,-5])
                return soln['fun']
            
            return brute(cost_given_theta, ([-.3,-.8],), Ns=6)[0]
            
        pool = mp.Pool(mp.cpu_count()-1)
        try:
            theta = np.array(pool.map(one_loop_wrapper, range(n_samples)))
        finally:
            pool.close()
        return theta

    @classmethod
    def std_of_mean(cls, x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
#end ThetaFitter



class SiteExtractor():
    def __init__(self,
                 event_type='battle',
                 dxdt=(320,128),
                 resolution_range=[640,320,160],
                 null_type=None):
        """
        Parameters
        ----------
        event_type : str, 'battle'
        dxdt : tuple, (320, 128)
            Resolution on which conflict avalanches are built
        resolution_range : list, [640,320,160]
            Length "dx" with which to pixelate inside of conflict avalanche
        null_type : str, None
            Possible values: 'shuffle'
        """

        self.eventType = event_type
        self.dxdt = dxdt
        self.resolutionRange = resolution_range  
        self.subdf, self.gridOfSplits = pipe.load_default_pickles(event_type=event_type)
        if not null_type is None and not null_type in ['shuffle']:
            raise NotImplentedError("Unrecognized null type.")
        else:
            self.nullType = 'shuffle'
            
    def run(self):
        """Run internal avalanche pixelation. This is the main function that one should
        call to run analysis.
        """

        one_cluster_wrapper = self.define_one_cluster_wrapper()

        # parallelize over each conflict cluster ==========================
        with threadpool_limits(limits=1, user_api='blas'):
            with mp.Pool(mp.cpu_count()-1) as pool:
                results = pool.map(one_cluster_wrapper, self.setup_mp_args())

        # combine events into a single dictionary
        eventsPerSite = {}  # indexed by resolution factor denominator
        for r in self.resolutionRange:
            eventsPerSite[r] = []

        for r in results:
            for k in r.keys():
                eventsPerSite[k].append(r[k])

        self.eventsPerSite = eventsPerSite
    
    def define_one_cluster_wrapper(self):
        """
        """
        def one_cluster_wrapper(args, resolutionRange=self.resolutionRange):
            """main loop
            """
            
            counter, heightBds, widthBds, xydata = args
            rng = np.random.RandomState()
            eventsLabeledByPixelIx = {}  # dict containing indices of events clustered per Voronoi site
            
            # generate coarse grid
            poissdCoarse = PoissonDiscSphere(np.pi/resolutionRange[0] * 3,
                                             height_bds=heightBds,
                                             width_bds=widthBds,
                                             iprint=False,
                                             rng=rng)
            poissdCoarse.sample()
            print("Coarse grid done for event %d."%counter)
            
            # generate grid
            poissd = PoissonDiscSphere(np.pi/resolutionRange[0],
                                       height_bds=heightBds,
                                       width_bds=widthBds,
                                       coarse_grid=poissdCoarse.samples,
                                       iprint=False,
                                       rng=rng)
            poissd.sample()

            # pixelate
            xydata = xydata.loc[:,['LONGITUDE','LATITUDE']].values.copy()
            xydata[:,0] %= 360
            # transform to radians
            xydata *= np.pi/180
            try:
                for i in range(len(resolutionRange)-1):
                    assert poissd.within_limits(xydata)
                    eventsLabeledByPixelIx[resolutionRange[i]] = poissd.pixelate(xydata)
                    poissd.expand(resolutionRange[i+1]/resolutionRange[i], force=True)
                # this contains the pixel indices to which each event belongs
                # these are given in the order of the data points provided in xydata
                assert poissd.within_limits(xydata)
                eventsLabeledByPixelIx[resolutionRange[len(resolutionRange)-1]] = poissd.pixelate(xydata)
            except NoVoronoiTilesRemaining:
                # if the voronoi pixelation runs out of points to coarse grain with,
                # put everything into the same pixel labeled '0'
                for i in range(len(resolutionRange)-1):
                    eventsLabeledByPixelIx[resolutionRange[i]] = [0]*len(xydata)
                eventsLabeledByPixelIx[resolutionRange[len(resolutionRange)-1]] = [0]*len(xydata)

            print("Done with event %d."%counter)
            return eventsLabeledByPixelIx

        return one_cluster_wrapper

    def setup_mp_args(self):
        """Generator for arguments to pass to multiprocess loop.
        
        Returns
        -------
        generator
            tuple of (counter, heightBds, widthBds, xydata)
            xydata is list pd.DataFrame for events that belong to each cluster
        """
        
        # This determines how much extra room there is around the xy bounds on the map. If
        # this is too small, you expect distortion in the Voronoi grid close to the
        # boundaries of the box.
        padding_in_radians = np.pi/self.resolutionRange[-1] * 3.1

        counter = 0
        for i,clustix in enumerate(self.gridOfSplits[self.dxdt]):
            uxy, counts = np.unique(self.subdf[['LONGITUDE','LATITUDE']].loc[clustix],
                                    axis=0,
                                    return_counts=True)
            dt = self.subdf['EVENT_DATE'].loc[clustix].max() - self.subdf['EVENT_DATE'].loc[clustix].min()
            
            if self.nullType=='shuffle':
                clustix = np.random.permutation(clustix)

            if (counts.size>=MIN_NO_GEO_PTS and  # only if there are sufficient unique locations
                len(clustix)>=4 and 
                int(dt/np.timedelta64(1,'D'))>=MIN_DAYS):
                # bounds for the Voronoi tiling
                # these need to be wide enough that multiple levels of resolution can be studied
                # this also needs to handle wrapping in the domain of latitude [-180,180] better 
                # but should work for now
                widthBds = ((uxy[:,0].min()/180*np.pi-padding_in_radians)%(2*np.pi),
                            (uxy[:,0].max()/180*np.pi+padding_in_radians)%(2*np.pi))
                heightBds = (uxy[:,1].min()/180*np.pi-padding_in_radians,
                             uxy[:,1].max()/180*np.pi+padding_in_radians)
                xydata = self.subdf.loc[clustix]
                
                yield counter, heightBds, widthBds, xydata
                counter += 1

    def cluster_ix(self):
        """Indices of clusters analyzed by coarse-graining function. This is for convenience when
        analyzing data after coarse-graining operation.
        
        Parameters
        ----------
        dxdt : tuple
            Specifying the length and time scales for binning with Voronoi tesselation.
            
        Returns
        -------
        list
        """
        
        clusterix = []
        counter = 0
        for i,clustix in enumerate(self.gridOfSplits[self.dxdt]):
            uxy, counts = np.unique(self.subdf[['LATITUDE','LONGITUDE']].loc[clustix],
                                    axis=0,
                                    return_counts=True)
            dt = self.subdf['EVENT_DATE'].loc[clustix].max() - self.subdf['EVENT_DATE'].loc[clustix].min()
            if (counts.size>=MIN_NO_GEO_PTS and # only if there are sufficient unique locations
                len(clustix)>=4 and 
                int(dt/np.timedelta64(1,'D'))>=MIN_DAYS):
                clusterix.append(i)

        if self.nullType:
            print("Null type shuffle will not be reflected in SiteExtractor.cluster_ix().")

        return clusterix
#end SiteExtractor
