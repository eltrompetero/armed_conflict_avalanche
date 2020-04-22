# ====================================================================================== #
# Module for conflict avalanche site dynamics.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
import numpy as np
from pyutils.acled_utils import *
import pyutils.new_pipeline as pipe
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
    """Class for fitting the peripheral suppression exponent theta on data.
    
    In order to use this class, must already know what gammaplus1 is beforehand. Using
    this value, one should solve for the best fit theta using solve_theta(). If a value
    for the vertical offset, c, is already known, as might be the case when calculating it
    explicitly from data, then this can be specified.
    """
    def __init__(self, trajectories, cluster_T, g,
                 gammaplus1=0.27,
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
        gammaplus : float, 0.27
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
        self.foundbinsix = [np.where(self.binsix==i)[0] for i in np.unique(self.binsix)]
    
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

        for counter,i in enumerate(np.unique(self.binsix)):
            avgdur[i] = np.mean([self.trajectories[ix].sum() / self.clusterT[ix]**d
                                 for ix in self.foundbinsix[counter]])
                                 #for ix in np.where(self.binsix==i)[0]])
            stddur[i] = self.std_of_mean([self.trajectories[ix].sum() / self.clusterT[ix]**d
                                          for ix in self.foundbinsix[counter]])
                                          #for ix in np.where(self.binsix==i)[0]])
        avgdur[avgdur==0] = np.nan

        return avgdur, stddur

    def solve_theta(self, **kwargs):
        """Find optimal value of theta using grid search algorithm.

        Parameters
        ----------
        c : float, None
        exclude_first : bool, False
            If True, don't fit to the first point. This is a good test of sensitivity of the fit.

        Returns
        -------
        float
            Theta
        tuple
            a, b, c
        """
        
        self.theta = brute(self.define_cost_given_theta(**kwargs), ([.4, 1.5],), Ns=10)[0]
        if 'c' in kwargs.keys():
            self.loga, self.logb = self.define_cost_given_theta(full_output=True, **kwargs)(self.theta)['x']
            self.c = kwargs['c']
        else:
            self.loga, self.logb, self.c = self.define_cost_given_theta(full_output=True,
                                                                        **kwargs)(self.theta)['x']
        return self.theta, (np.exp(self.loga), np.exp(self.logb), self.c)

    def define_cost_given_theta(self,
                                full_output=False,
                                **kwargs):
        """Calculate best fit for a given value of theta to find optimal theta
        that fits scaling collapse.

        See solve_theta().
        
        Parameters
        ----------
        weighted : bool, True
            Normalized error bars by uncertainty (more uncertainty means weaker
            constraint.
        full_output : bool, False
            If True, full result from scipy.optimize.minimize will be returned instead of
            just the function value.
        **kwargs

        Returns
        -------
        function
        """
        
        def theta_cost(theta):
            """
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

            Returns
            -------
            float
                Estimate of theta.
            """
            
            avgdur, stddur = self.avg_profile(theta)
            
            cost = self._define_cost_for_a_b_c(theta, avgdur, stddur, **kwargs)
            if 'c' in kwargs.keys():
                soln = minimize(cost, [3, -5], bounds=[(-np.inf,10),(-np.inf,10)])
            else:
                soln = minimize(cost, [3, -5, 10], bounds=[(-np.inf,10),(-np.inf,10),(0,np.inf)])

            if full_output:
                return soln
            return soln['fun']

        return theta_cost

    def _define_cost_for_a_b_c(self, theta, avgdur, stddur,
                               c=None,
                               exclude_first=False):
        if not c is None:
            if c<0 : raise Exception

            def cost(args,
                     theta=theta,
                     gammaplus1=self.gammaplus1,
                     binsx=self.binsx,
                     avgdur=avgdur,
                     stddur=stddur,
                     weighted=self.weighted,
                     c=c,
                     exclude_first=exclude_first):
                """Cost for finding coefficient a for scaling form of density of events 
                by scaled time.
                """
                
                loga, logb = args
                if loga>10 or logb>10 or theta<0 : return 1e30

                if weighted:
                    weights = stddur[1:]
                else:
                    weights = np.ones(avgdur.size-1)
                d = (((1 - binsx + np.exp(logb))**gammaplus1 * (np.exp(logb) + binsx)**-theta + c) * np.exp(loga) -
                     avgdur[1:])
                if exclude_first:
                    return  np.linalg.norm( d[1:] / weights[1:] ) + np.exp(logb)  # linear cost on large b
                else:
                    #return  np.linalg.norm( d / np.exp(loga) ) + np.exp(logb)  # linear cost on large b
                    return  np.linalg.norm( d / weights ) + np.exp(logb)  # linear cost on large b
        else:
            def cost(args,
                     theta=theta,
                     gammaplus1=self.gammaplus1,
                     binsx=self.binsx,
                     avgdur=avgdur,
                     stddur=stddur,
                     weighted=self.weighted,
                     exclude_first=exclude_first):
                """Cost for finding coefficient a for scaling form of density of events 
                by scaled time.
                """
                
                loga, logb, c = args
                if loga>10 or logb>10 or c<0 or theta<0 : return 1e30

                if weighted:
                    weights = stddur[1:]
                else:
                    weights = np.ones(avgdur.size-1)
                d = (((1 - binsx + np.exp(logb))**gammaplus1 * (np.exp(logb) + binsx)**-theta + c) * np.exp(loga) -
                     avgdur[1:])
                if exclude_first:
                    return  np.linalg.norm( d[1:] / weights[1:] ) + np.exp(logb)  # linear cost on large b
                else:
                    return  np.linalg.norm( d / weights ) + np.exp(logb)  # linear cost on large b
        return cost
    
    def boot_error_bars(self, n_sample):
        """Use bootstrap sampling to generate error bars on average profile.

        Parameters
        ----------
        n_sample : int

        Returns
        -------
        ndarray
            Error bars for plotting avgdur.
        """

        d = self.gammaplus1 - self.theta
        avgdur = self.avg_profile()[0]
        newavgdur = np.zeros((n_sample, avgdur.size))
        
        for counter in range(n_sample):
            # random sample
            randix = np.random.randint(self.g.size, size=self.g.size)
            g_ = self.g[randix]
            num = [self.trajectories[ix] for ix in randix]
            den = [self.clusterT[ix] for ix in randix]

            binsix_ = np.digitize(g_, self.bins)
            
            # calculate bootstrap sample average
            for i in np.unique(binsix_):
                newavgdur[counter,i] += np.mean([num[ix].sum() / den[ix]**d
                                                 for ix in np.where(binsix_==i)[0]])
        
        return np.vstack([avgdur - np.percentile(newavgdur, 5, axis=0),
                          np.percentile(newavgdur, 95, axis=0) - avgdur])

    def scaling_fun(self,
                    gammaplus1=None,
                    theta=None,
                    a=None,
                    b=None,
                    c=None):
         
        gammaplus1 = self.gammaplus1 if gammaplus1 is None else gammaplus1
        theta = self.theta if theta is None else theta
        a = np.exp(self.loga) if a is None else a
        b = np.exp(self.logb) if b is None else b
        c = self.c if c is None else c

        return lambda x, gp1=gammaplus1, th=theta, a=a, b=b, c=c: ((1 - x + b)**gammaplus1 *
                                                                   (b + x)**-theta + c) * a

    def plot(self, errs,
             theta_lb=None,
             theta_ub=None,
             var='r',
             return_handles=False):
        """Plot fit to data alongside given data.

        Parameters
        ----------
        stddur : ndarray
        errs : ndarray
        theta_lb : float, None
            This is plotted if both theta_lb and theta_ub are specified.
        theta_ub : float, None
        var : str, 'r'
            Variable that we're fitting in order to change the axis labels.
        return_handles : bool, False

        Returns
        -------
        mpl.Figure
        mpl.Axes
        list of line handles (optional)
        """
        
        avgdur, stddur = self.avg_profile()

        # setup
        x = np.linspace(.02, .98, 1000)
        h = []
        fig, ax = plt.subplots()
        
        # plot data and errorbars
        h.append( ax.errorbar(self.binsx, avgdur[1:], errs[:,1:], fmt='o') )
        
        # plot model
        h.append(ax.plot(x, self.scaling_fun()(x), 'k-')[0])
        if not theta_lb is None and not theta_ub is None:
            h.append(ax.plot(x, self.scaling_fun(theta=theta_lb)(x), 'k-.')[0])
            ax.plot(x, self.scaling_fun(theta=theta_ub)(x), 'k-.')
        h.append(ax.plot(x, self.scaling_fun(theta=0)(x), 'k--')[0])
        
        # plot properties
        ax.set(xlabel=r'relative starting time $g=t_0(x_i)/T$',
               ylabel=r'$\left\langle\ \overline{%s_{x_i}(T)/T^{1-\gamma_{%s}-\theta_{%s}}}\ \right\rangle$'%(var,var,var),
               xlim=(-.02,1.02),
               yscale='log')
        if not theta_lb is None and not theta_ub is None:
            ax.legend(h, 
                      ('Data',
                       r'$\theta_%s=%1.2f$'%(var, self.theta),
                       r'$\theta^+_%s$, $\theta^-_%s$'%(var, var),
                       r'$\theta_%s=0$'%var,
                       r'$\langle %s(t_0)\rangle$'%var),
                      fontsize='small', handlelength=1, loc=1, ncol=2, columnspacing=.55)
        else:
            ax.legend(h, 
                      ('Data',
                       r'$\theta_%s=%1.2f$'%(var, self.theta),
                       r'$\theta_%s=0$'%var,
                       r'$\langle %s(t_0)\rangle$'%var),
                      fontsize='small', handlelength=1, loc=1, ncol=2, columnspacing=.55)
        
        if return_handles: 
            return fig, ax, h
        return fig, ax

    def bootstrap(self, n_samples,
                  **solve_kw):
        """Bootstrap to get an estimate of error bars on theta.
        
        Parameters
        ----------
        n_samples : int
        **solve_kw

        Returns
        -------
        ndarray
            Estimates of theta.
        tuple of lists
            Other fit parameters.
        """

        theta = np.zeros(n_samples)
        
        def one_loop_wrapper(randcounter):
            np.random.seed()
            
            # create another instance of ThetaFitter given bootstrapped data
            randix = np.random.randint(self.g.size, size=self.g.size)
            tf = ThetaFitter([self.trajectories[i] for i in randix],
                             self.clusterT[randix],
                             self.g[randix],
                             gammaplus1=self.gammaplus1,
                             weighted=self.weighted)
            
            return tf.solve_theta(**solve_kw) 
        
        with mp.Pool(mp.cpu_count()-1) as pool:
            theta, abc = list(zip(*pool.map(one_loop_wrapper, range(n_samples))))
        return np.array(theta), list(zip(*abc))

    @classmethod
    def std_of_mean(cls, x):
        return np.std(x, ddof=1) / np.sqrt(len(x))
#end ThetaFitter


class ThetaFitterRBAC():
    """Fit theta for RBAC model results.
    """
    def __init__(self, gamma, bins, t0, t, rx):
        """
        Parameters
        ----------
        gamma : float
        g : ndarray
            Fractional time.
        t0 : ndarray
            Starting time per conflict site.
        t : ndarray
            Duration of entire conflict avalanche.
        rx : ndarray
            Total number of events per site.
        """

        self.g = (t0 + 1) / t
        self.binIx = np.digitize(self.g, bins) - 1
        self.gamma = gamma
        self.t = t
        self.rx = rx
        self.x = (bins[:-1] + bins[1:]) / 2

    def define_cost_a_b(self, theta, y, ystd,
                        exclude_k=0,
                        normalize_by_a=True):
        def cost_a_b(args, theta=theta, y=y, ystd=ystd):
            loga, logb = args
            
            if normalize_by_a:
                d = (np.exp(loga) * (1 - self.x + np.exp(logb))**(1-self.gamma) *
                     (self.x + np.exp(logb))**-theta - y) / np.exp(loga)
            else:
                d = (np.exp(loga) * (1 - self.x + np.exp(logb))**(1-self.gamma) *
                     (self.x + np.exp(logb))**-theta - y) / ystd
            
            # skip for the first k data points for fitting
            return np.linalg.norm(d[exclude_k:]) + np.exp(logb)
        return cost_a_b
    
    def define_cost(self, **kwargs):
        def cost(theta):
            rxNormalized = self.rx / self.t**(1 - self.gamma - theta)
            y = np.array([rxNormalized[i==self.binIx].mean() for i in range(self.x.size)])
            ystd = np.array([rxNormalized[i==self.binIx].std()/np.sqrt((i==self.binIx).sum())
                             for i in range(self.x.size)])
            cost_a_b = self.define_cost_a_b(theta, y, ystd, **kwargs)
            return minimize(cost_a_b, [np.log(y.mean()), -3])['fun']
        return cost

    def solve_theta(self, full_output=False, **kwargs):
        """
        Parameters
        ----------
        exclude_first : bool, False
        """

        # find theta
        fmin = lambda x, y, **kwargs: minimize(x, y, bounds=[(-.5,1.7)], **kwargs)
        # use grid search to find optimal theta value
        thetafit = brute(self.define_cost(**kwargs), [(.2,1.3)], Ns=10, finish=fmin)[0]
        
        # solve for corresponding a and logb given theta
        rxNormalized = self.rx / self.t**(1 - self.gamma - thetafit)
        y = np.array([rxNormalized[i==self.binIx].mean() for i in range(self.x.size)])
        ystd = np.array([rxNormalized[i==self.binIx].std()/np.sqrt((i==self.binIx).sum())
                         for i in range(self.x.size)])
        cost_a_b = self.define_cost_a_b(thetafit, y, ystd, **kwargs)
        soln = minimize(cost_a_b, [thetafit, self.gamma])
        logafit, logbfit = soln['x']
        
        if full_output:
            thetafit, np.exp(logafit), np.exp(logbfit), soln
        return thetafit, np.exp(logafit), np.exp(logbfit)
#end ThetaFitterRBAC



class SiteExtractor():
    """Identify conflict pixels.
    """
    def __init__(self,
                 event_type='battle',
                 dxdt=(320,128),
                 resolution_range=[640,320,160],
                 null_type=None,
                 rng=None):
        """
        Parameters
        ----------
        event_type : str, 'battle'
        dxdt : tuple, (320, 128)
            Resolution on which conflict avalanches are built to be fed into gridOfSplits.
        resolution_range : list, [640,320,160]
            Length "dx" with which to pixelate inside of conflict avalanche.
        null_type : str, None
            Possible values: 'shuffle'
        """

        self.eventType = event_type
        self.dxdt = dxdt
        self.resolutionRange = resolution_range  
        self.subdf, self.gridOfSplits = pipe.load_default_pickles(event_type=event_type)
        if not null_type is None and not null_type in ['shuffle']:
            raise NotImplentedError("Unrecognized null type.")
        elif null_type=='shuffle':
            self.nullType = 'shuffle'
        else:
            self.nullType = None
        self.rng = rng or np.random
            
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
        """This function will be fed into .run().
        """
        def one_cluster_wrapper(args, resolutionRange=self.resolutionRange):
            """main loop
            """
            
            counter, heightBds, widthBds, xydata, seed = args
            rng = np.random.RandomState(seed)
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
            
            # is this what I want when I shuffle?
            if self.nullType=='shuffle':
                clustix = self.rng.permutation(clustix)

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
                
                yield counter, heightBds, widthBds, xydata, self.rng.randint(0, 2**32-1)
                counter += 1

    def cluster_ix(self):
        """Indices of clusters analyzed by coarse-graining function. This is for convenience when
        analyzing data after coarse-graining operation.

        Note that this does not account for shuffling null.
        
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
