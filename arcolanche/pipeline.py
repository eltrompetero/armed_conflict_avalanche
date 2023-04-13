# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *
from .construct import Avalanche
from .data import ACLED2020
<<<<<<< HEAD
from workspace.utils import save_pickle, load_pickle
from multiprocess import Pool
from .construct import discretize_conflict_events
from itertools import product
from .analysis import ConflictZones
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy
import geopandas
from voronoi_globe.interface import load_voronoi
import random
=======
from workspace.utils import save_pickle
from itertools import product
from .analysis import ConflictZones
>>>>>>> 009e92ec2d088ecee9afdb170836fe4292b94d47



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

def generate_avalanches(conflict_type="battles"):
    """Generates causal conflict avalanches for a given conflict type.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Saves pickles of avalanches in both box and event form. 
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    time_list = [1,2,4,8,16,32,64,128,256,512]
    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    gridix_list = range(1,21)

    dx_time_gridix = list(product(dx_list,time_list,gridix_list))

    def looper(args):
        dx,time,gridix = args

        ava = Avalanche(time,dx,gridix,conflict_type=conflict_type)

        ava_box = [[tuple(i) for i in ava.time_series.loc[a].drop_duplicates()\
                    .values[:,::-1]] for a in ava.avalanches]
        ava_event = ava.avalanches

        discretize_conflict_events.cache_clear()

        path = f"avalanches/{conflict_type}/gridix_{gridix}/te"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["ava_box","ava_event"] ,\
                    f"avalanches/{conflict_type}/gridix_{gridix}/te/te_ava_{str(time)}_{str(dx)}.p" ,\
                    True)


    with Pool() as pool:
        pool.map(looper , dx_time_gridix)

def actor_similarity_generator(conflict_type):
    """Calculates actor similarity matrix for a given conflict type.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Saves pickles of actor similarity matrix.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    time_list = [1,2,4,8,16,32,64,128,256,512]
    threshold = [1]

    def actor_ratio_loop_wrapper(args):
        return ConflictZones(*args).similarity_score()
        
    for gridix in range(1,21):
        actor_similarity = np.zeros((len(dx_list),len(time_list)))

        dxdt = list(product(time_list,dx_list,threshold,[gridix],[conflict_type]))

        with Pool() as pool:
            output = list(pool.map(actor_ratio_loop_wrapper , dxdt))

        for i,(j,k) in zip(range(len(dxdt)),product(range(len(time_list)),range(len(dx_list)))):
            actor_similarity[k][j] = output[i]
        
        path = "mesoscale_data"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["actor_similarity"],f"mesoscale_data/similarity_matrix_{gridix}_{conflict_type}.p", True)

def data_used_generator(conflict_type):
    """Calculates data used matrix for a given conflict type.
    Only avalanches with number of spatial or temporal bin > 1 are
    considered valid and used in calculation.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Saves pickles of data used matrix.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    def data_used_wrapper(args):
        time,dx,gridix,conflict_type = args
        
        box_path = f"avalanches/{conflict_type}/gridix_{gridix}/te/te_ava_{str(time)}_{str(dx)}.p"
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]
        
        if(conflict_type == "battles"):
            ACLED_data = ACLED2020.battles_df()
        elif(conflict_type == "VAC"):
            ACLED_data = ACLED2020.vac_df()
        elif(conflict_type == "RP"):
            ACLED_data = ACLED2020.riots_and_protests_df()

        events_used = 0
        for num in range(len(ava_box)):
            if(len(ava_box[num]) != 1):
                events_used += len(ava_event[num])
        
        return events_used / len(ACLED_data)

    for gridix in range(1,21):
        dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
        time_list = [1,2,4,8,16,32,64,128,256,512]
        
        dxdt = list(product(time_list,dx_list,[gridix],[conflict_type]))
        
        data_used = np.zeros((len(dx_list),len(time_list)))
                
        with Pool() as pool:
            output = list(pool.map(data_used_wrapper , dxdt))
        
        for i,(j,k) in zip(range(len(dxdt)),product(range(len(time_list)),range(len(dx_list)))):
            data_used[k][j] = output[i]

        path = "mesoscale_data"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)        

        save_pickle(["data_used"],f"mesoscale_data/data_used_{gridix}_{conflict_type}.p", True)

def z_fine_calulator(values_matrix,filter_cutoff):
    interp = interpolation_scipy(values_matrix)
    
    t_range = np.arange(values_matrix.shape[1])
    x_range = np.arange(values_matrix.shape[0])
    
    t_range_fine = np.linspace(t_range[0],t_range[-1] , 1000)
    x_range_fine = np.linspace(x_range[0],x_range[-1],1000)
    z_fine  = interp(x_range_fine , t_range_fine)
    
    ix = (z_fine < filter_cutoff)
    
    z_fine[ix] = 0
    z_fine[z_fine != 0] = 1
    z_fine[z_fine == 0] = np.nan
    
    return z_fine , t_range_fine , x_range_fine

def z_fine_calulator_2(values_matrix):
    interp = interpolation_scipy(values_matrix)
    
    t_range = np.arange(values_matrix.shape[1])
    x_range = np.arange(values_matrix.shape[0])
    
    t_range_fine = np.linspace(t_range[0],t_range[-1] , 1000)
    x_range_fine = np.linspace(x_range[0],x_range[-1],1000)
    z_fine  = interp(x_range_fine , t_range_fine)
    
    return z_fine , t_range_fine , x_range_fine

def interpolation_scipy(values_matrix):
    t_range = np.arange(values_matrix.shape[1])
    x_range = np.arange(values_matrix.shape[0])
    
    interpolation = scipy.interpolate.RectBivariateSpline(x_range , t_range , values_matrix)
    
    return interpolation

def data_used_plot(conflict_type):
    """Plot contour plot of percentage of data used.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Displays the contour plot of percentage of data used.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    fig, ax = plt.subplots(figsize=(10,10))
    cb_ax = fig.add_axes([.12, -0.005, .785, .05])

    yaxis_label = r"length scale $b$ (km)"
    xaxis_label = r"time scale $a$ (days)"

    for gridix in [5]:
        load_pickle(f"mesoscale_data/data_used_{gridix}_{conflict_type}.p")

        z_fine_data , t_range_fine , x_range_fine = z_fine_calulator(data_used,0.75)

        a = np.where(z_fine_data==1)

        b = np.empty(z_fine_data.shape)
        b[:] = 0
        b[a] = 1

        cax = ax.imshow(data_used , interpolation="bicubic" , cmap=plt.cm.Blues, vmin=0 , vmax=1 , aspect="auto")
        contr = ax.contour(t_range_fine,x_range_fine, b , linewidths=5 , colors='w')

        cbar = fig.colorbar(cax, cax=cb_ax, orientation='horizontal', fraction=.2)
        cbar.ax.set_title(r"% data involved $\Phi$", pad=35 , fontsize=60, y=-5)
        cbar.ax.set_xticks([0,.5,1])
        cbar.ax.set_xticklabels(labels=[0,.5,1] ,fontsize=60)
        cbar.ax.tick_params(pad=15)

    positionsx = [0,3,6,9]
    labelsx = [1,8,64,512]


    positionsy = [0,4,8,12]
    labelsy = reversed([22,88,352,1408])

    ax.xaxis.labelpad = 15
    ax.set_xticklabels(labelsx, fontsize=60)
    ax.set_yticklabels(labelsy, fontsize=60)
    ax.set(xticks=positionsx,
           yticks=positionsy,
           xlabel=xaxis_label, ylabel=yaxis_label);

    ax.set_xlabel(xaxis_label, fontsize=60, labelpad=35)
    ax.set_ylabel(yaxis_label , fontsize=60, labelpad=33)
    ax.tick_params(axis='both', which='major', pad=30)
    ax.tick_params(axis='both', which='minor', pad=30)

    ax.set_xlim([0,9])
    ax.set_ylim([12,0])

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.tick_params(width=3,length=5)

def actor_similarity_plot(conflict_type):
    """Plot contour plot of actor similarity matrix contour.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Displays the contour plot of actor similarity matrix contour.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."
       
    fig, ax = plt.subplots(figsize=(10,10))
    cb_ax = fig.add_axes([.12, -0.005, .785, .05])
    yaxis_label = r"length scale $b$ (km)"
    xaxis_label = r"time scale $a$ (days)"

    for gridix in [3]:
        load_pickle(f"mesoscale_data/similarity_matrix_{gridix}_{conflict_type}.p")

        actor_similarity_threshold = (max(-np.log10(actor_similarity.flatten()))-min(-np.log10(actor_similarity.flatten()))) / 2

        z_fine_actor , t_range_fine , x_range_fine = z_fine_calulator(-np.log10(actor_similarity),actor_similarity_threshold)
        a = np.where(z_fine_actor==1)

        b = np.empty(z_fine_actor.shape)
        b[:] = 0
        b[a] = 1

        cax = ax.imshow(np.log10(actor_similarity) , interpolation="bicubic" , cmap=plt.cm.OrRd , aspect="auto",
                        vmin=np.log10(10**-2))
        contr = ax.contour(t_range_fine,x_range_fine,b , linewidths=5 , colors='w')

        cbar = fig.colorbar(cax, cax=cb_ax, orientation='horizontal', fraction=.2)
        cbar.set_ticks([0, -1, -2])
        cbar.ax.set_xticklabels([r'$10^0$',r'$10^{-1}$',r'$10^{-2}$'] , fontsize=60)
        cbar.ax.set_title(r"actor similarity $S$", pad=35 ,  fontsize=60 , y=-5)
        cbar.ax.tick_params(pad=15)


    positionsx = [0,3,6,9]
    labelsx = [1,8,64,512] 

    positionsy = [0,4,8,12]
    labelsy = reversed([22,88,352,1408])

    ax.xaxis.labelpad = 15
    ax.set(xticks=positionsx, xticklabels=labelsx,
           yticks=positionsy, yticklabels=labelsy,
           xlabel=xaxis_label,)
           #ylabel=r"spatial bin size (km)");

    ax.set_xlabel(xaxis_label, fontsize=60, labelpad=35)
    ax.set_ylabel(yaxis_label , fontsize=60, labelpad=33)
    ax.tick_params(axis='both', which='major', labelsize=60, pad=30)
    ax.tick_params(axis='both', which='minor', labelsize=60, pad=30)


    ax.set_xlim([0,9])
    ax.set_ylim([12,0])

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position('right')

    ax.tick_params(width=3,length=5)

def mesoscale_plot(conflict_type):
    """Plot mesoscale.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Displays the mesoscale.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    time_list = [1,2,4,8,16,32,64,128,256,512]

    fig, ax = plt.subplots(figsize=(10,10))

    initial_index = 0
    threshold_list = []
    for gridix in range(1,21):
        initial_index += 1

        load_pickle(f"mesoscale_data/similarity_matrix_{gridix}_{conflict_type}.p")
        load_pickle(f"mesoscale_data/data_used_{gridix}_{conflict_type}.p")

        actor_similarity_threshold = (max(-np.log10(actor_similarity.flatten()))-min(-np.log10(actor_similarity.flatten()))) / 2
        threshold_list.append(10**-actor_similarity_threshold)

        z_fine_data , t_range_fine , x_range_fine = z_fine_calulator(data_used,0.75)
        z_fine_actor , t_range_fine , x_range_fine = z_fine_calulator(-np.log10(actor_similarity),actor_similarity_threshold)

        a = np.where(np.logical_and(z_fine_actor==1,z_fine_data==1))

        b = np.empty(z_fine_actor.shape)
        b[:] = 0
        b[a] = 1



        if(initial_index == 1):
            weighted_contour = np.zeros(actor_similarity.shape)

            temp_matrix = np.zeros(actor_similarity.shape)
            temp_matrix[(-np.log10(actor_similarity) > actor_similarity_threshold) & (data_used > 0.75)] = 1

            weighted_contour += temp_matrix
        else:
            temp_matrix = np.zeros(actor_similarity.shape)
            temp_matrix[(-np.log10(actor_similarity) > actor_similarity_threshold) & (data_used > 0.75)] = 1

            weighted_contour += temp_matrix


    ax.imshow(np.flip(weighted_contour,axis=0), interpolation="bicubic",  cmap="Purples" , aspect="auto")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = 15

    plt.xlim([-0.5,len(time_list)-0.5])
    plt.ylim([-0.5,len(dx_list)-0.5])

    positionsy = [0,6,12]
    labelsy = [22,175,1450]
    positionsy = [0,4,8,12]
    labelsy = [22,87,349,1450]

    positionsx = [0,3,6,9]
    labelsx = ['$2^{0}$','$2^{3}$','$2^{6}$','$2^{9}$']
    labelsx = ['1','8','64','512']

    plt.xticks(positionsx , labelsx , fontsize=30)
    plt.yticks(positionsy , labelsy , fontsize=30)
    plt.xticks(positionsx , [] , fontsize=30)
    plt.yticks(positionsy , [] , fontsize=30)


    ax.tick_params(width=2,length=20)

    width = 0.3
    height = 0.4

    # top left
    box = Ellipse((0, 12), width=width , height=height, fc='red', ec='red', lw=2.5)
    ax.add_patch(box)
    # bottom left
    box = Ellipse((2, 2), width=width , height=height, fc='orange', ec='orange', lw=2.5)
    ax.add_patch(box)
    # top right
    box = Ellipse((7, 11), width=width , height=height, fc='blue', ec='blue', lw=2.5)
    ax.add_patch(box)
    # bottom right
    box = Ellipse((8, 1), width=width , height=height, fc='magenta', ec='magenta', lw=2.5)
    ax.add_patch(box)
    # main
    box = Ellipse((6, 4), width=width , height=height, fc='g', ec='g', lw=2.5)
    ax.add_patch(box)

    ax.patch.set_alpha(0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    t_range = np.arange((weighted_contour/20).shape[1])
    x_range = np.arange((weighted_contour/20).shape[0])

    a,b,c = z_fine_calulator_2((weighted_contour/20))
    contr = ax.contour(b,c,np.flip(a,axis=0) ,
                       linewidths=1 ,
                       levels=[0.5,0.9] ,
                       colors=["indigo","white"])
    ax.clabel(contr, inline=True, fontsize=20)

def set_ax(country):
    """Sets the area boundary around a specific country while plotting African map.

    Parameters
    ----------
    country : str

    Returns
    -------
    ndarray
    """

    if(country == "Nigeria"):
        return np.array([0.5535987755982988+330/180*np.pi, 
                         0.895006094968746698+330/180*np.pi,
                         0.0553981633974483, 
                         0.3203047484373349])*180/np.pi
    elif(country == "Somalia"):
        return np.array([1.135987755982988+330/180*np.pi,
                         1.4406094968746698+330/180*np.pi,
                         -0.033981633974483,
                         0.2203047484373349])*180/np.pi
    elif(country == "Egypt"):
        return np.array([0.9535987755982988+330/180*np.pi, 
                         1.156094968746698+330/180*np.pi,
                         0.370853981633974483, 
                         0.5603047484373349])*180/np.pi
    elif(country == "Sierra Leone"):
        return np.array([0.235987755982988+330/180*np.pi, 
                         0.4506094968746698+330/180*np.pi,
                         0.0553981633974483, 
                         0.2203047484373349])*180/np.pi
    elif(country == "Libya"):
        return np.array([0.3735987755982988+330/180*np.pi,
                         1.0406094968746698+330/180*np.pi,
                         0.3700853981633974483, 
                         0.7203047484373349])*180/np.pi
    elif(country == "Kenya"):
        return np.array([1.035987755982988+330/180*np.pi,
                         1.2806094968746698+330/180*np.pi,
                         -0.09281633974483,
                         0.1203047484373349])*180/np.pi
    elif(country == "Uganda"):
        return np.array([1.01987755982988+330/180*np.pi,
                         1.1506094968746698+330/180*np.pi,
                         -0.03281633974483,
                         0.0903047484373349])*180/np.pi
    elif(country == "Democratic Republic of the Congo"):
        return np.array([0.913755982988+330/180*np.pi,
                         1.1506094968746698+330/180*np.pi,
                         -0.1081633974483,
                         0.103047484373349])*180/np.pi
    else:
        return np.array([0.05235987755982988+330/180*np.pi, 
                      1.6406094968746698+330/180*np.pi,
                      -0.5853981633974483, 
                      0.6203047484373349])*180/np.pi
    
def conflict_zones_figure(time,dx,gridix,conflict_type="battles"):
    """Displays a figure of conflict zones at given time,dx and gridix
    for a given conflict type.
    
    Paramemters
    -----------
    time : int
    dx : int
    gridix : int
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Displays conflict zones.
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    polygons = load_voronoi(dx,gridix)

    acled_data = ACLED2020.battles_df()
    location_data_arr = np.array(acled_data[["LONGITUDE","LATITUDE"]])

    conflict_zones = ConflictZones(time,dx,threshold=1,gridix=gridix,conflict_type=conflict_type)

    zones = conflict_zones.generator()

    sorted_zones = sorted(zones , key=len)
    sorted_zones.reverse()

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[['continent', 'geometry']]

    continents = world.dissolve(by='continent')

    africa = gpd.GeoDataFrame(continents["geometry"]["Africa"] , geometry=0)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    for index in range(len(sorted_zones)):
        color_tuple = (random.uniform(0,1) , random.uniform(0,1) , random.uniform(0,1))
        if(len(sorted_zones[index]) < 5):
            gpd.clip(polygons.loc[sorted_zones[index]],continents["geometry"]["Africa"]) \
                .plot(ax=ax , facecolor=color_tuple , alpha=0.2)           
        else:
            gpd.clip(polygons.loc[sorted_zones[index]],continents["geometry"]["Africa"]) \
                .plot(ax=ax , facecolor=color_tuple)


    africa.plot(ax=ax , facecolor="none" , edgecolor="black" , linewidth=6)
    ax.set_extent(set_ax("Full"))
