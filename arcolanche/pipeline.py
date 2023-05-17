# ====================================================================================== #
# Module for pipelining revised analysis.
# Author: Eddie Lee, edlee@santafe.edu
#         Niraj Kushwaha, nirajkkushwaha1@gmail.com
# ====================================================================================== #
from .utils import *
from .construct import Avalanche
from .data import ACLED2020
from workspace.utils import save_pickle, load_pickle
from multiprocess import Pool,cpu_count
from itertools import product
from .analysis import ConflictZones
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy
import geopandas
from voronoi_globe.interface import load_voronoi, load_centers
from vincenty import vincenty
import random
from matplotlib.lines import Line2D
from collections import Counter
from mycolorpy import colorlist as mcp
import matplotlib
import itertools
import tqdm
from .construct import discretize_conflict_events


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

def generate_avalanches(conflict_type="battles" , num_threads=cpu_count()):
    """Generates causal conflict avalanches for a given conflict type.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    num_threads : int , multiprocess.cpu_count()
        Number of threads used to generate avalanches.
        Reduce the number of threads to reduce the RAM usage.
    
    Returns
    -------
    None
    
    Saves pickles of avalanches in both box and event form. 
    """

    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    time_list = [1,2,4,8,16,32,64,128,256,512]
    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    gridix_list = range(21,100)

    dx_time_gridix = list(product(dx_list,time_list,gridix_list))

    def looper(args):
        dx,time,gridix = args

        ava = Avalanche(time,dx,gridix,conflict_type=conflict_type)

        ava_box = [[tuple(i) for i in ava.time_series.loc[a].drop_duplicates()\
                    .values[:,::-1]] for a in ava.avalanches]
        ava_event = ava.avalanches

        #discretize_conflict_events.cache_clear()

        path = f"avalanches/{conflict_type}/gridix_{gridix}/te"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["ava_box","ava_event"] ,\
                    f"avalanches/{conflict_type}/gridix_{gridix}/te/te_ava_{str(time)}_{str(dx)}.p" ,\
                    True)

        return None

    # with Pool() as pool:
    #     pool.map(looper , dx_time_gridix)

    output = []
    pool = Pool(processes=num_threads, maxtasksperchild=1)
    print("Generating conflict avalanches:")
    for result in tqdm.tqdm(pool.imap(looper,dx_time_gridix) , total=len(dx_time_gridix)):
        output.append(result)

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
    
    print("Calculating actor overlap scores:")
    for gridix in tqdm.tqdm(range(1,100)):
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
    
    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    time_list = [1,2,4,8,16,32,64,128,256,512]

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

    print("Calculating data used percentages:")
    for gridix in tqdm.tqdm(range(1,100)):
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
    for gridix in range(1,100):
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


    t_range = np.arange((weighted_contour/initial_index).shape[1])
    x_range = np.arange((weighted_contour/initial_index).shape[0])

    a,b,c = z_fine_calulator_2((weighted_contour/initial_index))
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

def first_event_ava_country(ava_event , event_locations, country):
    """Returns all the avalanches whose first event was in a given country.
    
    Parameters
    ----------
    ava_event : list of lists
        list of avalanches in event form
    event_locations : dataframe
        location of each and every conflict event in the dataset
    country : str
    
    Returns
    -------
    list of lists
    """

    african_countries = gpd.read_file(f"data/africa_countries/afr_g2014_2013_0.shp")
    country_pol = african_countries[african_countries["ADM0_NAME"] == country]
    
    first_events = []
    for ava in ava_event:
        first_events.append(ava[0])
        
    first_events_locations = event_locations.loc[first_events]
    
    country_ava_index = []
    for i in range(len(first_events_locations)):
        in_or_not = country_pol["geometry"].iloc[0].contains(first_events_locations["geometry"].iloc[i])
        if(in_or_not == True):
             country_ava_index.append(i)
                
    country_first_events = list(first_events_locations.iloc[country_ava_index].index)
    
    country_avas = []
    for ava in ava_event:
        if(ava[0] in country_first_events):
            country_avas.append(ava)
            
    country_avas_sorted = sorted(country_avas , key=len)
    
    return country_avas_sorted

def container_avalanche_finder(ava_event , fixed_events):
    """Returns the avalanche which contains the given fixed events.
    
    Parameters
    ----------
    ava_event : list of list
        list of avalanches in event form
    fixed_events : list
        list of fixed events that must be part of the avalanche we need
        
    Returns
    -------
    list
        avalanche that contains all the fixed events
    """
    
    fixed_events_set = set(fixed_events)
    found = False
    for ava in ava_event:
        ava_set = set(ava)
        
        if(fixed_events_set.issubset(ava_set)):
            return ava
            found = True
            break
        else:
            pass
        
    if(found):
        pass
    else:
        raise Exception("Such container avalanche doesn't exist!")
    
def all_ava_country(ava_event , event_locations, country):
    """Returns all avalanches which contains atleast one coflict event in a 
    given country.
    
    Parameters
    ----------
    ava_event : list of lists
        list of avalanches in event form
    event_locations : dataframe
        location of each and every conflict event in the dataset
    country : str
    
    Returns
    -------
    list of lists
    """
    
    african_countries = gpd.read_file(f"data/africa_countries/afr_g2014_2013_0.shp")
    country_pol = african_countries[african_countries["ADM0_NAME"] == country]

    country_ava_index = []
    for index,ava in enumerate(ava_event):
        for event in ava:
            in_or_not = country_pol["geometry"].iloc[0].contains(event_locations["geometry"].loc[event])
            if(in_or_not == True):
                country_ava_index.append(index)
                break
    
    country_avas = np.array(ava_event , dtype=object)[country_ava_index].tolist()
    country_avas_sorted = sorted(country_avas , key=len)
    
    return country_avas_sorted

def conflict_clusters_figure():
    """Plots the conflict clusters' figure from the paper.
    """
    
    conflict_type = "battles"

    fig , axs = plt.subplots(3, 3 , subplot_kw={'projection': ccrs.PlateCarree()})

    fig.set_figheight(25)
    fig.set_figwidth(25)

    fig.tight_layout()


    acled_data = ACLED2020.battles_df()

    event_locations = acled_data[["LONGITUDE","LATITUDE"]]
    event_locations = gpd.GeoDataFrame(event_locations,
                                       geometry=gpd.points_from_xy(acled_data.LONGITUDE, acled_data.LATITUDE))
    event_locations_unique = gpd.GeoDataFrame(event_locations["geometry"].unique() , geometry=0)


    for i,j in product(range(3),range(3)):
        axs[i,j].add_feature(cfeature.COASTLINE, linewidth=1.5, alpha=0.5)
        axs[i,j].add_feature(cfeature.BORDERS, linewidth=1.5, alpha=0.5)
        #ax.add_feature(cfeature.LAND)
        axs[i,j].add_feature(cfeature.OCEAN,alpha=0.8)
        axs[i,j].add_feature(cfeature.RIVERS)
        event_locations_unique.plot(ax=axs[i,j], facecolor='lightgray', edgecolor='lightgray' , marker="." , linewidth=0.2 ,
                                    alpha=1)


    fontsize = 35
    font = {'size': fontsize}
    fig.text(0.29,0.69,"(a)",fontdict=font)
    fig.text(0.62,0.69,"(b)",fontdict=font)
    fig.text(0.95,0.69,"(c)",fontdict=font)
    fig.text(0.29,0.39,"(d)",fontdict=font)
    fig.text(0.62,0.39,"(e)",fontdict=font)
    fig.text(0.95,0.39,"(f)",fontdict=font)
    fig.text(0.29,0.11,"(g)",fontdict=font)
    fig.text(0.62,0.11,"(h)",fontdict=font)
    fig.text(0.95,0.11,"(i)",fontdict=font)



    axs[0,1].arrow(17,6, -2.1,1.7, width=1 , head_length=2 , head_width=2 , facecolor="black" , edgecolor="black",
                  zorder=100)
    axs[1,1].arrow(44.5,9.5, 2,0.4, width=1 , head_length=1.8 , head_width=2 , facecolor="black" , edgecolor="black",
                  zorder=101)
    axs[1,1].arrow(36,-1, 2,0.2, width=1 , head_length=2 , head_width=2 , facecolor="black" , edgecolor="black",
                  zorder=102)
    axs[2,1].arrow(-6.8,10.6, -1.6,-1.6, width=0.75 , head_length=1.1 , head_width=1.6 , facecolor="black" , edgecolor="black",
                  zorder=103)


    fontsize = 52
    pad = 25
    font_label = {'size': fontsize}
    axs[0,0].set_title("heuristic clusters" , fontdict=font_label , pad=pad)
    axs[0,1].set_title("conflict avalanches" , fontdict=font_label , pad=pad)
    axs[0,2].set_title("interaction zones" , fontdict=font_label , pad=pad)

    axs[0,0].text(-0.05, 0.55, 'Nigeria', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor', fontsize=fontsize,
            transform=axs[0,0].transAxes)
    axs[1,0].text(-0.05, 0.55, 'Somalia', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor', fontsize=fontsize,
            transform=axs[1,0].transAxes)
    axs[2,0].text(-0.05, 0.55, 'Sierra Leone', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor', fontsize=fontsize,
            transform=axs[2,0].transAxes)






    ####### Nigeria #######
    time = dt = 64
    dx = 320
    gridix = 3
    type_of_algo = "te"

    country = "Nigeria"

    axs[0,0].set_extent(set_ax(country))
    axs[0,1].set_extent(set_ax(country))
    axs[0,2].set_extent(set_ax(country))

    #### Using our method ####


    box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                        f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
    with open(box_path,"rb") as f:
        ava = pickle.load(f)
    ava_box = ava["ava_box"]
    ava_event = ava["ava_event"]

    country_avas_sorted = first_event_ava_country(ava_event , event_locations , country)


    color_list = ["red" , "orange" , "purple" , "blue" , "brown"]
    marker_list = ["." , "X" , "v" , "^"]
    linewidth_list = [0.3,0.2,0.05,0.05]


    for i in [1,2,3]:
        event_locations.loc[country_avas_sorted[-i]].plot(ax=axs[0,1], facecolor=color_list[i-1],
                                                           edgecolor=color_list[i-1] , marker=".",
                                                           alpha=1 , linewidth=0.8)

    separatist_events = [27000,26981]
    ava_to_plot = container_avalanche_finder(ava_event , separatist_events)
    event_locations.loc[ava_to_plot].plot(ax=axs[0,1], facecolor="forestgreen",
                                                       edgecolor="forestgreen" , marker=".",
                                                       alpha=1 , linewidth=0.8)


    ##### Using Actor names ####

    if(country == "Nigeria"):
        actor_name_list = ["Boko Haram" , "Fulani", "PDP" , "Ambazonian Separatists"]

    include_notes = False


    actor1_list = acled_data["ACTOR1"].to_list()
    actor2_list = acled_data["ACTOR2"].to_list()
    notes = acled_data["NOTES"].to_list()

    actor_clusters = []
    for actor_name in actor_name_list:
        actor1_indexes = [index for actor,index in zip(actor1_list,acled_data.index)if actor_name in actor]
        actor2_indexes = [index for actor,index in zip(actor2_list,acled_data.index)if actor_name in actor]
        actor_index_from_notes = [index for note,index in zip(notes,acled_data.index)if actor_name in str(note)]

        if(include_notes):
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes + actor_index_from_notes)))
        else:
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes)))

        actor_clusters.append(event_cluster)



    color_list = ["red" , "green" , "orange" , "blue"]

    for index,actor_cluster in enumerate(actor_clusters):
        event_locations.loc[actor_cluster].plot(ax=axs[0,0], 
                                                 facecolor=color_list[index],
                                                 edgecolor=color_list[index],
                                                 marker=".", 
                                                 linewidth=0.8)



    if(country == "Nigeria"):
        actor_name_list = ["BH" , "Fulani" , "PDP" , "AS"]

    legend_elements = []
    for i in range(len(actor_name_list)):
        legend_elements.append(Line2D([0],[0],marker='o',color="white",label=actor_name_list[i],
                                      markerfacecolor=color_list[i],
                                      markersize=15))



    axs[0,0].legend(handles=legend_elements , framealpha=1 , fontsize=30 ,
               borderpad=0.5,
               labelspacing=0.4,
              borderaxespad=0.1,
              columnspacing=0.2,
              frameon=False,
              shadow=False,
              handlelength=0)

    #### Probability thing ####

    boko_haram_events = [134916,134972,134975,134976]
    separatist_events = [27000]  # Earlier event 26981 was also in this list
    zamfara_events = [145797]

    if(country == "Nigeria"):
        fixed_events = boko_haram_events

    ava_to_plot_list = []
    for gridix in range(1,100):
        box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]

        ava_to_plot = container_avalanche_finder(ava_event , fixed_events)
        ava_to_plot_list.append(ava_to_plot)

    events_combined = list(itertools.chain.from_iterable(ava_to_plot_list))
    event_certainty_dict = dict(Counter(events_combined))

    event_certainty_dict = {k: round(v / len(ava_to_plot_list),1) for k, v in event_certainty_dict.items()}

    colormap = "autumn_r"
    color1= np.array(mcp.gen_color(cmap=colormap,n=24))
    color1 = color1[4:24:2]


    for i in range(1,11):
        current_events = [key for key,value in event_certainty_dict.items() if value == i/10]

        event_locations.loc[current_events]. \
                plot(ax=axs[0,2], facecolor=color1[i-1], edgecolor=color1[i-1] , marker="." , alpha=1 , linewidth=0.8)


    cmapnew = matplotlib.colors.ListedColormap(color1)
    img = plt.imshow(np.array([[1,2,3,4,5,6,7,8,9]])/9, cmap=cmapnew)
    img.set_visible(False)


    legend_elements = []
    for i in [2,9]:
        current_events = [key for key,value in event_certainty_dict.items() if value >= i/10]
        a = gpd.GeoDataFrame({"date":[""] , "geometry":[event_locations.loc[current_events].unary_union.convex_hull]} , geometry="geometry")
        b = gpd.GeoDataFrame(pd.concat([event_locations.loc[current_events] , a], ignore_index=True))

        b.iloc[-1:].plot(ax=axs[0,2] , alpha=1 , facecolor="none" , edgecolor=color1[i-1] , linewidth=2)

        legend_elements.append(Line2D([0], [0], color=color1[i-1], lw=4, label=f"$p={i/10}$"))


    axs[0,2].legend(handles=legend_elements , framealpha=1 , fontsize=30 ,
               borderpad=0.5,
               labelspacing=0.4,
              borderaxespad=0.1,
              columnspacing=0.2,
              frameon=False,
              shadow=False,
              handlelength=2)



    ##### Other sphere of influences ####

    if(country == "Nigeria"):
        fixed_events = separatist_events

    colormap = "BuGn"
    color1= np.array(mcp.gen_color(cmap=colormap,n=24))
    color1 = color1[4:24:2]

    ava_to_plot_list = []
    for gridix in range(1,100):
        box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]

        ava_to_plot = container_avalanche_finder(ava_event , fixed_events)
        ava_to_plot_list.append(ava_to_plot)

    events_combined = list(itertools.chain.from_iterable(ava_to_plot_list))
    event_certainty_dict = dict(Counter(events_combined))

    event_certainty_dict = {k: round(v / len(ava_to_plot_list),1) for k, v in event_certainty_dict.items()}

    for i in range(5,11):
        current_events = [key for key,value in event_certainty_dict.items() if value == i/10]

        event_locations.loc[current_events]. \
                plot(ax=axs[0,2], facecolor=color1[i-1], edgecolor=color1[i-1] , marker="." , alpha=1 , linewidth=0.8)


    for i in [5]:
        current_events = [key for key,value in event_certainty_dict.items() if value >= i/10]
        a = gpd.GeoDataFrame({"date":[""] , "geometry":[event_locations.loc[current_events].unary_union.convex_hull]} , geometry="geometry")
        b = gpd.GeoDataFrame(pd.concat([event_locations.loc[current_events] , a], ignore_index=True))

        b.iloc[-1:].plot(ax=axs[0,2] , alpha=1 , facecolor="none" , edgecolor=color1[i-1] , linewidth=2)

        legend_elements.append(Line2D([0], [0], color=color1[i-1], lw=4, label=f"$p={i/10}$"))


    if(country == "Nigeria"):
        fixed_events = zamfara_events

    colormap = "Purples"
    color1= np.array(mcp.gen_color(cmap=colormap,n=24))
    color1 = color1[4:24:2]

    ava_to_plot_list = []
    for gridix in range(1,100):
        box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]

        ava_to_plot = container_avalanche_finder(ava_event , fixed_events)
        ava_to_plot_list.append(ava_to_plot)

    events_combined = list(itertools.chain.from_iterable(ava_to_plot_list))
    event_certainty_dict = dict(Counter(events_combined))

    event_certainty_dict = {k: round(v / len(ava_to_plot_list),1) for k, v in event_certainty_dict.items()}

    for i in range(5,11):
        current_events = [key for key,value in event_certainty_dict.items() if value == i/10]

        event_locations.loc[current_events]. \
                plot(ax=axs[0,2], facecolor=color1[i-1], edgecolor=color1[i-1] , marker="." , alpha=1 , linewidth=0.8)


    for i in [5]:
        current_events = [key for key,value in event_certainty_dict.items() if value >= i/10]
        a = gpd.GeoDataFrame({"date":[""] , "geometry":[event_locations.loc[current_events].unary_union.convex_hull]} , geometry="geometry")
        b = gpd.GeoDataFrame(pd.concat([event_locations.loc[current_events] , a], ignore_index=True))

        b.iloc[-1:].plot(ax=axs[0,2] , alpha=1 , facecolor="none" , edgecolor=color1[i-1] , linewidth=2)

        legend_elements.append(Line2D([0], [0], color=color1[i-1], lw=4, label=f"$p={i/10}$"))



    ####### Somalia ########  
    time = dt = 32
    dx = 320
    gridix = 3
    type_of_algo = "te"

    country = "Somalia"

    axs[1,0].set_extent(set_ax(country))
    axs[1,2].set_extent(set_ax(country))
    axs[1,1].set_extent(set_ax(country))

    #### Using our method ####


    box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                        f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
    with open(box_path,"rb") as f:
        ava = pickle.load(f)
    ava_box = ava["ava_box"]
    ava_event = ava["ava_event"]

    country_avas_sorted = first_event_ava_country(ava_event , event_locations , country)

    color_list = ["red" , "blue" , "forestgreen" , "orange" , "brown" , "black","orange"]
    marker_list = ["." , "X" , "v" , "^"]
    linewidth_list = [0.3,0.2,0.05,0.05]

    for i in [1,2,3,7]:
        event_locations.loc[country_avas_sorted[-i]].plot(ax=axs[1,1], facecolor=color_list[i-1],
                                                           edgecolor=color_list[i-1] , marker=".",
                                                           alpha=1 , linewidth=0.8)




    ##### Using Actor names ####


    if(country == "Somalia"):
        actor_name_list = ["Al Shabaab" , "ASWJ" , "Islamic Courts Union"]

    include_notes = False


    actor1_list = acled_data["ACTOR1"].to_list()
    actor2_list = acled_data["ACTOR2"].to_list()
    notes = acled_data["NOTES"].to_list()

    actor_clusters = []
    for actor_name in actor_name_list:
        actor1_indexes = [index for actor,index in zip(actor1_list,acled_data.index)if actor_name in actor]
        actor2_indexes = [index for actor,index in zip(actor2_list,acled_data.index)if actor_name in actor]
        actor_index_from_notes = [index for note,index in zip(notes,acled_data.index)if actor_name in str(note)]

        if(include_notes):
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes + actor_index_from_notes)))
        else:
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes)))

        actor_clusters.append(event_cluster)


    color_list = ["red" , "green" , "blue" , "orange"]

    for index,actor_cluster in enumerate(actor_clusters):
        event_locations.loc[actor_cluster].plot(ax=axs[1,0], 
                                                 facecolor=color_list[index],
                                                 edgecolor=color_list[index],
                                                 marker=".", 
                                                 linewidth=0.8)



    if(country == "Somalia"):
        actor_name_list = ["Al-Shabaab" , "ASWJ" , "ICU"]


    legend_elements = []
    for i in range(len(actor_name_list)):
        legend_elements.append(Line2D([0],[0],marker='o',color=color_list[i],label=actor_name_list[i],
                                      markerfacecolor=color_list[i],
                                      markersize=15))


    axs[1,0].legend(handles=legend_elements , framealpha=1 , fontsize=30 ,
               borderpad=0.5,
               labelspacing=0.4,
              borderaxespad=0.1,
              columnspacing=0.2,
              frameon=False,
              shadow=False,
              handlelength=0)


    #### Probability thing ####

    somalia_events = [169883] # 167530,167535,167557

    if(country == "Somalia"):
        fixed_events = somalia_events


    ava_to_plot_list = []
    for gridix in range(1,100):
        box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]

        ava_to_plot = container_avalanche_finder(ava_event , fixed_events)
        ava_to_plot_list.append(ava_to_plot)

    events_combined = list(itertools.chain.from_iterable(ava_to_plot_list))
    event_certainty_dict = dict(Counter(events_combined))

    event_certainty_dict = {k: round(v / len(ava_to_plot_list),1) for k, v in event_certainty_dict.items()}

    colormap = "autumn_r"
    color1= np.array(mcp.gen_color(cmap=colormap,n=24))
    color1 = color1[4:24:2]

    for i in range(1,11):
        current_events = [key for key,value in event_certainty_dict.items() if value == i/10]

        event_locations.loc[current_events]. \
                plot(ax=axs[1,2], facecolor=color1[i-1], edgecolor=color1[i-1] , marker="." , alpha=1 , linewidth=0.8)


    cmapnew = matplotlib.colors.ListedColormap(color1)
    img = plt.imshow(np.array([[1,2,3,4,5,6,7,8,9]])/9, cmap=cmapnew)
    img.set_visible(False)

    legend_elements = []
    for i in [2,9]:
        current_events = [key for key,value in event_certainty_dict.items() if value >= i/10]
        a = gpd.GeoDataFrame({"date":[""] , "geometry":[event_locations.loc[current_events].unary_union.convex_hull]} , geometry="geometry")
        b = gpd.GeoDataFrame(pd.concat([event_locations.loc[current_events] , a], ignore_index=True))

        b.iloc[-1:].plot(ax=axs[1,2] , alpha=1 , facecolor="none" , edgecolor=color1[i-1] , linewidth=2)

        legend_elements.append(Line2D([0], [0], color=color1[i-1], lw=4, label=f"$p={i/10}$"))



    ######## Sierra Leone #######
    time = dt = 64
    dx = 453
    gridix = 3
    type_of_algo = "te"

    country = "Sierra Leone"

    axs[2,2].set_extent(set_ax(country))
    axs[2,0].set_extent(set_ax(country))
    axs[2,1].set_extent(set_ax(country))

    #### Using our method ####

    box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                        f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
    with open(box_path,"rb") as f:
        ava = pickle.load(f)
    ava_box = ava["ava_box"]
    ava_event = ava["ava_event"]

    country_avas_sorted = first_event_ava_country(ava_event , event_locations , country)


    color_list = ["red" , "blue" , "orange" , "forestgreen" , "brown"]
    marker_list = ["." , "X" , "v" , "^"]
    linewidth_list = [0.3,0.2,0.05,0.05]


    for i in [1,2,3,4]:
        event_locations.loc[country_avas_sorted[-i]].plot(ax=axs[2,1], facecolor=color_list[i-1],
                                                           edgecolor=color_list[i-1] , marker=".",
                                                           alpha=1 , linewidth=0.8)



    ##### Using Actor names ####


    if(country == "Sierra Leone"):
        actor_name_list = ["RUF" , "LURD"]


    include_notes = False


    actor1_list = acled_data["ACTOR1"].to_list()
    actor2_list = acled_data["ACTOR2"].to_list()
    notes = acled_data["NOTES"].to_list()

    actor_clusters = []
    for actor_name in actor_name_list:
        actor1_indexes = [index for actor,index in zip(actor1_list,acled_data.index)if actor_name in actor]
        actor2_indexes = [index for actor,index in zip(actor2_list,acled_data.index)if actor_name in actor]
        actor_index_from_notes = [index for note,index in zip(notes,acled_data.index)if actor_name in str(note)]

        if(include_notes):
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes + actor_index_from_notes)))
        else:
            event_cluster = sorted(list(set(actor1_indexes + actor2_indexes)))

        actor_clusters.append(event_cluster)




    color_list = ["red" , "blue" , "forestgreen" , "orange"]

    for index,actor_cluster in enumerate(actor_clusters):
        event_locations.loc[actor_cluster].plot(ax=axs[2,0], 
                                                 facecolor=color_list[index],
                                                 edgecolor=color_list[index],
                                                 marker=".", 
                                                 linewidth=0.8)




    if(country == "Sierra Leone"):
        actor_name_list = ["RUF" , "LURD"]

    legend_elements = []
    for i in range(len(actor_name_list)):
        legend_elements.append(Line2D([0],[0],marker='o',color=color_list[i],label=actor_name_list[i],
                                      markerfacecolor=color_list[i],
                                      markersize=15))


    axs[2,0].legend(handles=legend_elements , framealpha=1 , fontsize=30 ,
               borderpad=0.5,
               labelspacing=0.4,
              borderaxespad=0.5,
              columnspacing=0.2,
              frameon=False,
              shadow=False,
              handlelength=0)


    #### Probability thing ####

    sl_events = [161317]

    if(country == "Sierra Leone"):
        fixed_events = sl_events

    ava_to_plot_list = []
    for gridix in range(1,100):
        box_path = (f"avalanches/{conflict_type}/gridix_{gridix}/{type_of_algo}/" +
                            f"{type_of_algo}_ava_{str(dt)}_{str(dx)}.p")
        with open(box_path,"rb") as f:
            ava = pickle.load(f)
        ava_box = ava["ava_box"]
        ava_event = ava["ava_event"]

        ava_to_plot = container_avalanche_finder(ava_event , fixed_events)
        ava_to_plot_list.append(ava_to_plot)

    events_combined = list(itertools.chain.from_iterable(ava_to_plot_list))
    event_certainty_dict = dict(Counter(events_combined))

    event_certainty_dict = {k: round(v / len(ava_to_plot_list),1) for k, v in event_certainty_dict.items()}

    colormap = "autumn_r"
    color1= np.array(mcp.gen_color(cmap=colormap,n=24))
    color1 = color1[4:24:2]

    for i in range(1,11):
        current_events = [key for key,value in event_certainty_dict.items() if value == i/10]

        event_locations.loc[current_events]. \
                plot(ax=axs[2,2], facecolor=color1[i-1], edgecolor=color1[i-1] , marker="." , alpha=1 , linewidth=0.8)


    cmapnew = matplotlib.colors.ListedColormap(color1)
    img = plt.imshow(np.array([[1,2,3,4,5,6,7,8,9]])/9, cmap=cmapnew)
    img.set_visible(False)

    legend_elements = []
    for i in [2,9]:
        current_events = [key for key,value in event_certainty_dict.items() if value >= i/10]
        a = gpd.GeoDataFrame({"date":[""] , "geometry":[event_locations.loc[current_events].unary_union.convex_hull]} , geometry="geometry")
        b = gpd.GeoDataFrame(pd.concat([event_locations.loc[current_events] , a], ignore_index=True))

        b.iloc[-1:].plot(ax=axs[2,2] , alpha=1 , facecolor="none" , edgecolor=color1[i-1] , linewidth=2)

        legend_elements.append(Line2D([0], [0], color=color1[i-1], lw=4, label=f"$p={i/10}$"))

def discretize_conflict_events_using_centers(dt, dx, gridix=0, conflict_type='battles', year_range=False):
    """Merged GeoDataFrame for conflict events of a certain type into the Voronoi
    cells. Time discretized.

    Parameters
    ----------
    dt : int
    dx : int
    gridix : int, 0
    conflict_type : str, 'battles'
    year_range : tuple, False

    Returns
    -------
    GeoDataFrame
        New columns 't' and 'x' indicate time and Voronoi bin indices.
    """
    
    def inverse_transform(coordinates):
        """From lon, lat to angles coordinates accounting for the longitudinal shift necessary
        to get to Africa.
        This is an inverse to transform function in voronoi_globe.utils.transform().

        Parameters
        ----------
        event_coordinates : ndarray

        Returns
        -------
        ndarray
            angles.
        """

        coordinates[:,0] -= 330
        coordinates_angles = (coordinates/180) * np.pi

        return coordinates_angles
    

    load_pickle(f"voronoi_grids/{dx}/{str(gridix).zfill(2)}.p")
    global centers
    centers = load_centers(dx,gridix)
    

    def potential_polygons_extractor(event_index):
        return poissd.neighbors(event_coordinates_angles[event_index] , apply_dist_threshold=True)   
    
    def enclosing_polygon_number(args):
        (event_location,neighbor_indexes) = args

        neighbor_distances = []
        for neighbor_index in neighbor_indexes:
            potential_nearest_center = (centers[neighbor_index].x , centers[neighbor_index].y)

            neighbor_distances.append(vincenty(potential_nearest_center,event_location))

        return neighbor_indexes[neighbor_distances.index(min(neighbor_distances))]


    if(conflict_type == "battles"):
        df = ACLED2020.battles_df(to_lower=True,year_range=year_range)
    elif(conflict_type == "RP"):
        df = ACLED2020.riots_and_protests_df(to_lower=True,year_range=year_range)
    elif(conflict_type == "VAC"):
        df = ACLED2020.vac_df(to_lower=True,year_range=year_range)

    conflict_ev = gpd.GeoDataFrame(df[['event_date','longitude','latitude']],
                                   geometry=gpd.points_from_xy(df.longitude, df.latitude),
                                   crs="EPSG:4326")
    conflict_ev['t'] = (conflict_ev['event_date']-conflict_ev['event_date'].min()) // np.timedelta64(dt,'D')


    event_coordinates = np.column_stack((conflict_ev.geometry.apply(lambda x:x.x).to_list(),conflict_ev.geometry.apply(lambda x:x.y).to_list()))
    event_coordinates_angles = inverse_transform(event_coordinates.copy())

    with Pool() as pool:
        neighbors_list = pool.map(potential_polygons_extractor , range(len(conflict_ev)))
        output = pool.map(enclosing_polygon_number , zip(event_coordinates,neighbors_list))

    conflict_ev["x"] = output
    
    return conflict_ev

def conflict_dataframe_generator(conflict_type="battles"):
    """Generates GeodataFrames of conflict for all combinations of dx,dt and gridix
    for a given conflict type.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Saves pickles of conflict dataframes.
    """
    
    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    time_list = [1,2,4,8,16,32,64,128,256,512]
    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    gridix_list = range(21,99)

    dx_time_gridix = list(product(dx_list,time_list,gridix_list))

    for args in dx_time_gridix:
        dx,time,gridix = args
        print(dx,time,gridix)

        conflict_ev = discretize_conflict_events_using_centers(time,dx,gridix,conflict_type=conflict_type)

        path = f"avalanches/{conflict_type}/gridix_{gridix}/te"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["conflict_ev"] ,\
                    f"avalanches/{conflict_type}/gridix_{gridix}/te/conflict_ev_{str(time)}_{str(dx)}.p" ,\
                    True)

def center_neighbors_generator(conflict_type="battles"):
    """Generates Geodatafram of centers and their immediate neighbors for a given
    dx and gridix.
    
    Parameters
    ----------
    conflict_type : str , "battles"
    
    Returns
    -------
    None
    
    Saves pickles of polygon/center neighbors.
    """
    
    def inverse_transform(coordinates):
        """From lon, lat to angles coordinates accounting for the longitudinal shift necessary
        to get to Africa.
        This is an inverse to transform function in voronoi_globe.utils.transform().

        Parameters
        ----------
        event_coordinates : ndarray

        Returns
        -------
        ndarray
            angles.
        """

        coordinates[:,0] -= 330
        coordinates_angles = (coordinates/180) * np.pi

        return coordinates_angles

    def looper(args):
        dx , gridix = args

        load_pickle(f"voronoi_grids/{dx}/{str(gridix).zfill(2)}.p")
        centers = load_centers(dx,gridix)

        centers_arr = np.array([[point.x,point.y] for point in centers])
        centers_arr_transformed = inverse_transform(centers_arr)

        neighbors = []
        for index,center_transformed in enumerate(centers_arr_transformed):
            neighbors.append(sorted(poissd.neighbors(center_transformed , apply_dist_threshold=True)))
            neighbors[index].remove(index)

        polygons = gpd.GeoDataFrame(centers,geometry=0).rename_geometry("geometry")
        polygons["neighbors"] = neighbors

        path = f"avalanches/{conflict_type}/gridix_{gridix}"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["polygons"] ,\
                    f"avalanches/{conflict_type}/gridix_{gridix}/polygons_{str(dx)}.p" ,\
                    True)

        return None


    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    gridix_list = range(21,100)

    dx_gridix = list(product(dx_list,gridix_list))

    output = []
    pool = Pool()
    print("Generating polygon/center neighbors:")
    for result in tqdm.tqdm(pool.imap(looper,dx_gridix) , total=len(dx_gridix)):
        output.append(result)

def conflict_ev_generator(conflict_type="battles"):
    """Generates conflict_ev dataframes for all combinations of dx,dt and gridix
    for a given conflict type.
    
    Parameter
    ---------
    conflict_type : str, "battles"
        Choose amongst 'battles', 'VAC', and 'RP'.
    
    Returns
    -------
    None
    
    Saves pickles of conflict dataframes.
    """
    
    assert conflict_type in ['battles', 'VAC', 'RP'], "Non-existent conflict type."

    time_list = [1,2,4,8,16,32,64,128,256,512]
    dx_list = [20,28,40,57,80,113,160,226,320,453,640,905,1280]
    gridix_list = range(21,99)

    dx_time_gridix = list(product(dx_list,time_list,gridix_list))

    def looper(args):
        dx,time,gridix = args

        conflict_ev = discretize_conflict_events(time, dx, gridix, conflict_type, year_range=False)

        path = f"avalanches/{conflict_type}/gridix_{gridix}/te"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        save_pickle(["conflict_ev"] ,\
                    f"avalanches/{conflict_type}/gridix_{gridix}/te/conflict_ev_{str(time)}_{str(dx)}.p" ,\
                    True)

        return None

    # with Pool() as pool:
    #     pool.map(looper , dx_time_gridix)

    output = []
    pool = Pool()
    print("Generating conflict_ev files:")
    for result in tqdm.tqdm(pool.imap(looper,dx_time_gridix) , total=len(dx_time_gridix)):
        output.append(result)

    pool.close()