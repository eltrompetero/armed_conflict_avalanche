# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from .construct import Avalanche
from .data import ACLED2020
from workspace.utils import save_pickle
from itertools import product
from .analysis import ConflictZones



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
        return (np.np.percentile(r,2.5), np.np.percentile(r,97.5)), (fig,ax)
    return np.np.percentile(r,2.5), np.np.percentile(r,97.5)

def scaling_relations(dtdx=(64,320), gridix=3):
    """Prepare cache files for power law scaling, dynamical scaling, and exponent
    relations check.
    """

    from misc.stats import DiscretePowerLaw, PowerLaw, loglog_fit
    from voronoi_globe import max_geodist_pair, latlon2angle
    from warnings import catch_warnings, simplefilter
    
    # load data
    conf_df = ACLED2020.battles_df(to_lower=True)

    ava = Avalanche(*dtdx, gridix=gridix)

    # extract scaling variables
    R = np.array([len(i) for i in ava.avalanches])
    F = np.array([sum([conf_df['fatalities'].loc[i] for i in a]) for a in ava.avalanches])
    N = np.array([np.unique(ava.time_series['x'].loc[a]).size for a in ava.avalanches])
    T = np.array([(conf_df['event_date'].loc[a].max()-conf_df['event_date'].loc[a].min()).days
                   for a in ava.avalanches])

    L = []
    for a in ava.avalanches:
        latlon = np.unique(conf_df[['latitude','longitude']].loc[a].values, axis=0)
        if len(latlon)>1:
            L.append(max_geodist_pair(latlon2angle(latlon),
                                      return_dist=True)[1])
        else:
            L.append(0)
    L = np.array(L) * 6371

    # fit power laws
    pl_params = {}
    pl_params['F'] = DiscretePowerLaw.max_likelihood(F[F>1], lower_bound_range=(2, 100))
    pl_params['R'] = DiscretePowerLaw.max_likelihood(R[R>1], lower_bound_range=(2, 100))
    pl_params['N'] = DiscretePowerLaw.max_likelihood(N[N>1], lower_bound_range=(2, 100))
    pl_params['T'] = DiscretePowerLaw.max_likelihood(T[T>1], lower_bound_range=(2, 100))
    pl_params['L'] = PowerLaw.max_likelihood(L[L>0], lower_bound_range=(1e2, 1e3))
    
    # fit dynamical scaling exponents
    dyn_params = {}
    ix = (T>=pl_params['T'][1]) & (F>=pl_params['F'][1])
    dyn_params['F'] = loglog_fit(T[ix], F[ix])
    ix = (T>=pl_params['T'][1]) & (R>=pl_params['R'][1])
    dyn_params['R'] = loglog_fit(T[ix], R[ix])
    ix = (T>=pl_params['T'][1]) & (N>=pl_params['N'][1])
    dyn_params['N'] = loglog_fit(T[ix], N[ix])
    ix = (T>=pl_params['T'][1]) & (L>=pl_params['L'][1])
    dyn_params['L'] = loglog_fit(T[ix], L[ix])
    
    # check exponent relations
    exp_relations = {}
    exp_relations['F'] = pl_params['T'][0] - 1, dyn_params['F'][0] * (pl_params['F'][0]-1)
    exp_relations['R'] = pl_params['T'][0] - 1, dyn_params['R'][0] * (pl_params['R'][0]-1)
    exp_relations['N'] = pl_params['T'][0] - 1, dyn_params['N'][0] * (pl_params['N'][0]-1)
    exp_relations['L'] = pl_params['T'][0] - 1, dyn_params['L'][0] * (pl_params['L'][0]-1) 
    
    # compute fit errs
    errs = {}
    dpl = DiscretePowerLaw(*pl_params['F'])
    ks = dpl.ksval(F[F>=pl_params['F'][1]])

    F_above = F[F>=pl_params['F'][1]]
    F_below = F[(F>1)&(F<pl_params['F'][1])]
    pval, ks_sample, (alpha, lb) = dpl.clauset_test(F_above, ks,
                                                    lower_bound_range=(2,100), 
                                                    samples_below_cutoff=F_below,
                                                    return_all=True)

    errs['F'] = pval, alpha.std(), (np.percentile(alpha, 5), np.percentile(alpha, 95))

    dpl = DiscretePowerLaw(*pl_params['R'])
    ks = dpl.ksval(R[R>=pl_params['R'][1]])

    R_above = R[R>=pl_params['R'][1]]
    R_below = R[(R>1)&(R<pl_params['R'][1])]
    pval, ks_sample, (alpha, lb) = dpl.clauset_test(R_above, ks,
                                                    lower_bound_range=(2,100), 
                                                    samples_below_cutoff=R_below,
                                                    return_all=True)

    errs['R'] = pval, alpha.std(), (np.percentile(alpha, 5), np.percentile(alpha, 95))

    dpl = DiscretePowerLaw(*pl_params['N'])
    ks = dpl.ksval(N[N>=pl_params['N'][1]])

    N_above = N[N>=pl_params['N'][1]]
    N_below = N[(N>1)&(N<pl_params['N'][1])]
    pval, ks_sample, (alpha, lb) = dpl.clauset_test(N_above, ks,
                                                    lower_bound_range=(2,100), 
                                                    samples_below_cutoff=N_below,
                                                    return_all=True)

    errs['N'] = pval, alpha.std(), (np.percentile(alpha, 5), np.percentile(alpha, 95))

    dpl = DiscretePowerLaw(*pl_params['T'])
    ks = dpl.ksval(T[T>=pl_params['T'][1]])

    T_above = T[T>=pl_params['T'][1]]
    T_below = T[(T>1)&(T<pl_params['T'][1])]
    pval, ks_sample, (alpha, lb) = dpl.clauset_test(T_above, ks,
                                                    lower_bound_range=(2,200), 
                                                    samples_below_cutoff=T_below,
                                                    return_all=True)

    errs['T'] = pval, alpha.std(), (np.percentile(alpha, 5), np.percentile(alpha, 95))

    dpl = PowerLaw(*pl_params['L'])
    ks = dpl.ksval(L[L>=pl_params['L'][1]])

    L_above = L[L>=pl_params['L'][1]]
    L_below = L[(L>0)&(L<pl_params['L'][1])]
    with catch_warnings(record=True):
        simplefilter('always')
        pval, ks_sample, (alpha, lb) = dpl.clauset_test(L_above, ks,
                                                        lower_bound_range=(1e2,1e3), 
                                                        samples_below_cutoff=L_below,
                                                        return_all=True)

    errs['L'] = pval, alpha.std(), (np.percentile(alpha, 5), np.percentile(alpha, 95))
    
    # calculate exponent bounds error rel_err_bars
    rel_err_bars = []

    ix = (T>=pl_params['T'][1]) & (F>=pl_params['F'][1])
    samp = []
    T_ = T[ix]
    F_ = F[ix]
    for i in range(1000):
        randix = np.random.randint(T_.size, size=T_.size)
        samp.append(loglog_fit(T_[randix], F_[randix])[0])

    bds = exponent_bounds(errs['T'][2], errs['F'][2],
                          (np.percentile(samp, 5), np.percentile(samp, 95)))[0]
    rel_err_bars.append((exp_relations['F'][1] + 1 - bds[0], bds[1] - exp_relations['F'][1] - 1))

    ix = (T>=pl_params['T'][1]) & (R>=pl_params['R'][1])
    samp = []
    T_ = T[ix]
    R_ = R[ix]
    for i in range(1000):
        randix = np.random.randint(T_.size, size=T_.size)
        samp.append(loglog_fit(T_[randix], R_[randix])[0])

    bds = exponent_bounds(errs['T'][2], errs['R'][2],
                          (np.percentile(samp, 5), np.percentile(samp, 95)))[0]
    rel_err_bars.append((exp_relations['R'][1] + 1 - bds[0], bds[1] - exp_relations['R'][1] - 1))

    ix = (T>=pl_params['T'][1]) & (N>=pl_params['N'][1])
    samp = []
    T_ = T[ix]
    N_ = N[ix]
    for i in range(1000):
        randix = np.random.randint(T_.size, size=T_.size)
        samp.append(loglog_fit(T_[randix], N_[randix])[0])

    bds = exponent_bounds(errs['T'][2], errs['N'][2],
                          (np.percentile(samp, 5), np.percentile(samp, 95)))[0]
    rel_err_bars.append((exp_relations['N'][1] + 1 - bds[0], bds[1] - exp_relations['N'][1] - 1))

    ix = (T>=pl_params['T'][1]) & (L>=pl_params['L'][1])
    samp = []
    T_ = T[ix]
    L_ = L[ix]
    for i in range(1000):
        randix = np.random.randint(T_.size, size=T_.size)
        samp.append(loglog_fit(T_[randix], L_[randix])[0])

    bds = exponent_bounds(errs['T'][2], errs['L'][2],
                          (np.percentile(samp, 5), np.percentile(samp, 95)))[0]
    rel_err_bars.append((exp_relations['L'][1] + 1 - bds[0], bds[1] - exp_relations['L'][1] - 1))
    rel_err_bars = np.vstack(rel_err_bars).T

    save_pickle(['F','R','N','L','T','errs','pl_params','exp_relations','rel_err_bars','dyn_params'],
                f'cache/scaling_relations_{dtdx[0]}_{dtdx[1]}_{gridix}.p', True)

def similarity_score(conflict_type='battles'):
    """Averaged similarity matrix M across conflict zones. Saved to
    './cache/similarity_score_{gridix}_{conflict_type}.'

    Parameters
    ----------
    conflict_type : str, 'battles'
        Choose amongst 'battles', 'VAC', and 'RP'.
    """
    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    time_list = [1,2,4,8,16,32,64,128,256,512]
    threshold = 1

    for gridix in range(1, 21):
        actors_ratio = np.zeros((len(dx_list), len(time_list)))
         
        dxdt = list(product(time_list, dx_list, [threshold], [gridix], conflict_type))

        def actor_ratio_loop_wrapper(args):
            return ConflictZones(*args).similarity_score()
        
        with threadpool_limits(limits=1, user_api='blas'):
            with Pool() as pool:
                output = list(pool.map(actor_ratio_loop_wrapper, dxdt))
        
        for i, (j,k) in zip(range(len(dxdt)), product(range(len(time_list)), range(len(dx_list)))):
            actors_ratio[k][j] = output[i]
            
        save_pickle(['actors_ratio'], f'./cache/similarity_score_{gridix}_{conflict_type}.p', True)
