# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *



def rate_dynamics(dxdt=((160,16), (160,32), (160,64), (160,128), (160,256))):
    """Extract exponents from rate dynamical profiles.

    Automating analysis in "2019-11-18 rate dynamics.ipynb".

    Parameters
    ----------
    dxdt : tuple, ((160,16), (160,32), (160,64), (160,128), (160,256))

    Returns
    -------
    ndarray
        Dynamical exponent d_S/z.
    ndarray
        Dynamical exponent d_F/z.
    ndarray
        Dynamical exponent 1/z.
    """

    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist
    from .rate_dynamics import extract_from_df, interp_clusters

    dsz = np.zeros(len(dxdt))
    dfz = np.zeros(len(dxdt))
    invz = np.zeros(len(dxdt))
    subdf, gridOfSplits = load_default_pickles()

    for counter,dxdt_ in enumerate(dxdt):
        clusters = extract_from_df(subdf, gridOfSplits[dxdt_])

        # logarithmically spaced bins
        xinterp = np.linspace(0, 1, 100)
        bins = 2**np.arange(2, 14)
        npoints = np.arange(3, 20)[:bins.size-1]
        data, traj, clusterix = interp_clusters(clusters, xinterp, bins, npoints)


        offset = int(np.log2(dxdt_[1]))-2  # number of small bins to skip (skipping all avalanches with
                                           # duration <a days)

        # use variance in log space
        # these aren't weighted by number of data points because that returns something
        # similar and is more complicated
        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['S'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dsz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['F'][i+offset][1]/bins[i+offset]**a)
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        dfz[counter] = minimize(cost, 1.)['x']+1

        def cost(a):
            if a<=0 or a>5: return 1e30
            
            y = []
            for i,n in enumerate(npoints[offset:-1]):
                y.append(traj['L'][i+offset][1]/bins[i+offset]**a)
                
            y = np.log(np.vstack(y))
            return np.nansum(np.nanvar(y,0))
        invz[counter] = minimize(cost, .5)['x']
        
        print("dxdt = ",dxdt_)
        print('ds/z = %1.2f'%dsz[counter])
        print('df/z = %1.2f'%dfz[counter])
        print('1/z = %1.2f'%invz[counter])
        print()
    
    return dsz, dfz, invz

def loglog_fit_err_bars(x, y, fit_params, show_plot=False):
    """Calculate posterior probability of exponent parameter.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    fit_params : twople
    posterior : bool, False
    show_plot : bool, False

    Returns
    -------
    twople
        95% confidence intervals on exponent parameter assuming fixed offset.
    """

    from numpy import log
    from misc.stats import loglog_fit

    # posterior probability estimation of error bars
    fit_params=loglog_fit(x, y)

    resx=log(y) - np.polyval(fit_params, log(x))
    resy=(log(y) - fit_params[1])/fit_params[0] - log(x)
    varerr=np.concatenate((resx, resy)).var(ddof=1)

    def f(s, t=fit_params[1], x=x, y=y):
        """Function for calculating log likelihood."""
        return -1/2/varerr * ( ((log(y) - s*log(x) - t)**2 + ((log(y)-t)/s - log(x))**2).mean() )
    f=np.vectorize(f)

    # find bounding interval corresponding to a drop of exp(10) in probability
    dx=1e-2  # amount to increase bounds by per iteration
    bds=[fit_params[0], fit_params[0]]
    peak=f(fit_params[0])

    while (peak-f(bds[0]))<10:
        bds[0]-=dx
        
    while (peak-f(bds[1]))<10:
        bds[1]+=dx
    
    # construct discrete approximation to probability distribution
    x=np.linspace(*bds, 10_000)
    y=f(x)
    y-=y.max()
    p=np.exp(y)
    p/=p.sum()

    if show_plot:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        ax.plot(x, p)
        ax.vlines(fit_params[0], 0, p.max())
        ax.set(xlabel=r'$x$')
        ax.legend((r'$p(x)$', r'$s^*$'))

    # sample for confidence intervals
    r=np.random.choice(x, p=p, size=1_000_000)

    if show_plot:
        return (np.percentile(r,2.5), np.percentile(r,97.5)), (fig,ax)
    return np.percentile(r,2.5), np.percentile(r,97.5)

