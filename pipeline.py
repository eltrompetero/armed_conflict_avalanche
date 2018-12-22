# =============================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #
from .acled_utils import *
from data_sets.acled import *
import pickle
from statsmodels.distributions import ECDF
import dill
from misc.powerlaw import discrete_powerlaw_correction_spline, powerlaw_correction_spline

def check_consistency(eventType, gridno, pval_threshold=.05):
    """Check for which time and length scales the exponent relations between fatalities and sizes is consisten
    with the durations. These relations are consistent if they overlap within 90% confidence intervals.
    
    Also run a significance test to check which power law fits are significant at all. These are measured to
    p values of 0.1 which is the Clauset, Shalizi, Newman limit.

    Parameters
    ----------
    eventType : str
    gridno : int

    Returns
    -------
    bool ndarray
        True values indicate for which combination of length and time scales the exponent relations are
        consistent.
    bool ndarray
        True values indicate for which combination of length and time scales the exponent relations are
        both consistent and significant.
    """

    from .acled_utils import percentile_bds
    data = pickle.load(open('plotting/%s_ecdfs_inf_range%s.p'%(eventType,str(gridno).zfill(2)),'rb'))
    diameterInfo=data['diameterInfo']
    sizeInfo=data['sizeInfo']
    fatalityInfo=data['fatalityInfo']
    durationInfo=data['durationInfo']

    perc=(5,95)
    diameterInfo['nuBds']=np.array([percentile_bds(X, perc) for X in diameterInfo['nuSample']])
    sizeInfo['tauBds']=np.array([percentile_bds(X, perc) for X in sizeInfo['tauSample']])
    fatalityInfo['upsBds']=np.array([percentile_bds(X, perc) for X in fatalityInfo['upsSample']])
    durationInfo['alphaBds']=np.array([percentile_bds(X, perc) for X in durationInfo['alphaSample']])

    data=pickle.load(open('cache/%s_fractal_dimension%s.p'%(eventType,str(gridno).zfill(2)), 'rb'))
    dfGrid=data['dfGrid']
    dsGrid=data['dsGrid']
    dlGrid=data['dlGrid']
    
    # check fatality & duration
    consistent = np.zeros(len(sizeInfo['tau']), dtype=bool)==1
    for ix in range(len(sizeInfo['tau'])):
        if not np.isnan(durationInfo['alpha'][ix]): 
            consistent[ix] = check_relation(durationInfo['alphaBds'][ix], fatalityInfo['upsBds'][ix], dfGrid[0][4:])
    fatalityInfo['consistent'] = consistent
    
    # check size & duration
    consistent = np.zeros(len(sizeInfo['tau']), dtype=bool)==1
    for ix in range(len(sizeInfo['tau'])):
        if not np.isnan(durationInfo['alpha'][ix]):
            consistent[ix] = check_relation(durationInfo['alphaBds'][ix], sizeInfo['tauBds'][ix], dsGrid[0][4:])
    sizeInfo['consistent'] = consistent

    # check length & duration
    consistent = np.zeros(len(diameterInfo['nu']), dtype=bool)==1
    for ix in range(len(diameterInfo['nu'])):
        if not np.isnan(durationInfo['alpha'][ix]):
            consistent[ix] = check_relation(durationInfo['alphaBds'][ix], diameterInfo['nuBds'][ix], dlGrid[0][4:])
    diameterInfo['consistent'] = consistent

    # places in array where exponent relations are consistent
    consistent=sizeInfo['consistent'] & fatalityInfo['consistent']
    # places in array where exponent relation is consistent and measured power laws are significant
    sig = ( (sizeInfo['pval']>pval_threshold) &
            (fatalityInfo['pval']>pval_threshold) &
            (durationInfo['pval']>pval_threshold) )
        
    Lconsistent = consistent & diameterInfo['consistent']
    Lsig = (diameterInfo['pval']>pval_threshold) & (durationInfo['pval']>pval_threshold)
    Lsig &= sig
    
    return consistent, sig, Lconsistent, Lsig

def fractal_dimension(diameters, sizes, fatalities, durations, spaceThreshold, dayThreshold,
                      fitn=7, offset=-1, eventType=None, gridno=None):
    """Calculate fractal dimensions from scaling of means."""
    from .exponents import fractal_dimension, fractal_dimension_error

    T=len(dayThreshold)
    L=len(spaceThreshold)

    # Collect all (dx,dt) pairs to look at
    dsGrid=np.zeros((2,len(spaceThreshold)))
    dfGrid=np.zeros((2,len(spaceThreshold)))
    dlGrid=np.zeros((2,len(spaceThreshold)))

    # Collect all (dx,dt) pairs to look at
    dsGridBds=np.zeros((2,len(spaceThreshold),2))
    dfGridBds=np.zeros((2,len(spaceThreshold),2))
    dlGridBds=np.zeros((2,len(spaceThreshold),2))

    for i in range(len(spaceThreshold)):
        t=[t[t>1] for t in durations[i*T:(i+1)*T]]
        d=[d[d>0] for d in diameters[i*T:(i+1)*T]]
        f=[f[f>1] for f in fatalities[i*T:(i+1)*T]]
        s=[s[s>1] for s in sizes[i*T:(i+1)*T]]
            
        # calculate fractal dimension
        dfGrid[0,i], dfGridBds[0,i,:]=fractal_dimension(t[:fitn], f[:fitn])
        if offset==0:
            dfGrid[1,i], dfGridBds[1,i,:]=fractal_dimension(t[-fitn:], f[-fitn:])
        else:
            dfGrid[1,i], dfGridBds[1,i,:]=fractal_dimension(t[-fitn+offset:offset], f[-fitn+offset:offset])
        
        dsGrid[0,i], dsGridBds[0,i,:]=fractal_dimension(t[:fitn], s[:fitn])
        if offset==0:
            dsGrid[1,i], dsGridBds[1,i,:]=fractal_dimension(t[-fitn:], s[-fitn:])
        else:
            dsGrid[1,i], dsGridBds[1,i,:]=fractal_dimension(t[-fitn+offset:offset], s[-fitn+offset:offset])
            
        dlGrid[0,i], dlGridBds[0,i,:]=fractal_dimension(t[:fitn], d[:fitn])
        if offset==0:
            dlGrid[1,i], dlGridBds[1,i,:]=fractal_dimension(t[-fitn:], d[-fitn:])
        else:
            dlGrid[1,i], dlGridBds[1,i,:]=fractal_dimension(t[-fitn+offset:offset], d[-fitn+offset:offset])
    
    if not (eventType is None and gridno is None):
        pickle.dump({'dlGrid':dlGrid,'dfGrid':dfGrid,'dsGrid':dsGrid,
                     'dlGridBds':dlGridBds,'dlGridBds':dlGridBds,'dsGridBds':dsGridBds},
                    open('cache/%s_fractal_dimension%s.p'%(eventType,str(gridno).zfill(2)),'wb'),-1)
    return dlGrid, dlGridBds, dsGrid, dsGridBds, dfGrid, dfGridBds

def power_law_fit(eventType,
                  gridno,
                  diameters,
                  sizes,
                  fatalities,
                  durations,
                  finiteBound,
                  nBootSamples,
                  nCpus,
                  save_pickle=True,
                  run_diameter=True,
                  run_size=True,
                  run_fatality=True,
                  run_duration=True):
    """Run fitting and significance testing and pickle results.
    
    Parameters
    ----------
    eventType : str
    gridno : int
    diameters : list
    sizes : list
    fatalities : list
    durations : list
    finiteBound : bool
    nBootSamples : int
    nCpus : int
    save_pickle : bool, True
    run_diameter : bool, True
    run_size : bool, True
    run_fatality : bool, True
    run_duration : bool, True

    Returns
    -------
    str
        Filename where pickle is.
    dict, optional
        diameterInfo
    dict, optional
        sizeInfo
    dict, optional
        fatalityInfo
    dict, optional
        durationInfo
    """
    
    fname=('plotting/%s_ecdfs%s.p'%(eventType,str(gridno).zfill(2)) if finiteBound else
           'plotting/%s_ecdfs_inf_range%s.p'%(eventType,str(gridno).zfill(2)))
    
    diameterInfo={}
    sizeInfo={}
    durationInfo={}
    fatalityInfo={}
    
    if run_diameter:
        print("Starting diameter fitting...")
        upperBound = max([d.max() for d in diameters]) if finiteBound else np.inf
        output = _power_law_fit(diameters,
                                np.vstack([(d[d>0].min(),min(d.max()/2,1000)) for d in diameters]),
                                upperBound,
                                discrete=False,
                                n_boot_samples=nBootSamples,
                                n_cpus=nCpus)
        (diameterInfo['nu'],
         diameterInfo['nu1'],
         diameterInfo['lb'],
         diameterInfo['cdfs'],
         diameterInfo['ecdfs'],
         diameterInfo['fullecdfs'],
         diameterInfo['ksval'],
         diameterInfo['ksSample'],
         diameterInfo['pval'],
         diameterInfo['nuSample'],
         diameterInfo['lbSample']) = output
    
    if run_size:
        print("Starting size fitting...")
        upperBound = max([s.max() for s in sizes]) if finiteBound else np.inf
        output = _power_law_fit(sizes,
                                np.vstack([(s[s>1].min(),min(s.max()//10,1000)) for s in sizes]),
                                upperBound,
                                n_boot_samples=nBootSamples,
                                n_cpus=nCpus)
        (sizeInfo['tau'],
         sizeInfo['tau1'],
         sizeInfo['lb'],
         sizeInfo['cdfs'],
         sizeInfo['ecdfs'],
         sizeInfo['fullecdfs'],
         sizeInfo['ksval'],
         sizeInfo['ksSample'],
         sizeInfo['pval'],
         sizeInfo['tauSample'],
         sizeInfo['lbSample']) = output
    
    if run_fatality:
        print("Starting fatality fitting...")
        upperBound = max([f.max() for f in fatalities]) if finiteBound else np.inf
        output = _power_law_fit(fatalities,
                                  np.vstack([(f[f>1].min(),min(f.max()//10,1000)) for f in fatalities]),
                                  upperBound,
                                  n_boot_samples=nBootSamples,
                                  n_cpus=nCpus)
        (fatalityInfo['ups'],
         fatalityInfo['ups1'],
         fatalityInfo['lb'],
         fatalityInfo['cdfs'],
         fatalityInfo['ecdfs'],
         fatalityInfo['fullecdfs'],
         fatalityInfo['ksval'],
         fatalityInfo['ksSample'],
         fatalityInfo['pval'],
         fatalityInfo['upsSample'],
         fatalityInfo['lbSample']) = output
    
    if run_duration:
        print("Starting duration fitting...")
        upperBound = max([t.max() for t in durations]) if finiteBound else np.inf
        output = _power_law_fit(durations,
                                np.vstack([(t[t>1].min(),min(t.max()//10,1000)) for t in durations]),
                                upperBound,
                                n_boot_samples=nBootSamples,
                                n_cpus=nCpus)
        (durationInfo['alpha'],
         durationInfo['alpha1'],
         durationInfo['lb'],
         durationInfo['cdfs'],
         durationInfo['ecdfs'],
         durationInfo['fullecdfs'],
         durationInfo['ksval'],
         durationInfo['ksSample'],
         durationInfo['pval'],
         durationInfo['alphaSample'],
         durationInfo['lbSample']) = output
    
    if save_pickle:
        dill.dump({'diameterInfo':diameterInfo, 'sizeInfo':sizeInfo,
                   'durationInfo':durationInfo, 'fatalityInfo':fatalityInfo,
                   'nBootSamples':nBootSamples},
                    open(fname,'wb'),-1)
    
        return fname
    return fname, diameterInfo, sizeInfo, fatalityInfo, durationInfo

def _power_law_fit(Y, lower_bound_range, upper_bound,
		   discrete=True,
		   n_boot_samples=2500,
		   ksval_threshold=1.,
		   min_data_length=10,
		   n_cpus=None):
    """Pipeline max likelihood and mean scaling power law fits to conflict statistics. These are typically
    given by a coarse graining.

    Parameters
    ----------
    Y : list
    lower_bound_range : list of duples
    upper_bound : int
    discrete : bool, True
    n_boot_sample : int, 2500
        Default value gives accuracy of about 0.01.
    ksval_threshold : float, 1.
    min_data_length : int, 10
        Number of data points required before fitting process is initiated.
    n_cpus : int, None
    
    Returns
    -------
    ndarray
        Max likelihood exponent estimates.
    ndarray
        Lower bound estimates.
    ndarray
        Mean scalling exponent estimates. These are largely biased except for very heavy tailed distributions.
    ndarray
        KS statistic
    ndarray
        Sample of KS statistics.
    ndarray
        p-values.
    ndarray
        Samples of alpha from significance testing.
    ndarray
        Samples of lower bound from significance testing.
    """

    from multiprocess import Pool, cpu_count
    from misc.stats import PowerLaw, DiscretePowerLaw
    
    assert len(lower_bound_range)==len(Y)
    n_cpus = n_cpus or cpu_count()-1
    
    def f(args):
        i,y = args
        if discrete:
            y=y[y>1].astype(int)
        else:
            y=y[y>0]
        
        # don't do any calculation for distributions that are too small, all one value, or don't show much 
        # dynamic range
        if len(y)<min_data_length or (y[0]==y).all() or lower_bound_range[i][0]>(lower_bound_range[i][1]/2):
            return (np.nan, np.nan, np.nan, np.nan, None, np.nan, None, None)
        
        #return alpha, lb, alpha1, ksval, ksSample, pval, alphaSample, lbSample
        if discrete:
            alpha, lb = DiscretePowerLaw.max_likelihood(y,
                                                        lower_bound_range=lower_bound_range[i],
                                                        initial_guess=1.2,
                                                        upper_bound=upper_bound,
                                                        n_cpus=1)
            correction = discrete_powerlaw_correction_spline()
            alpha += correction(alpha, (y>=lb).sum(), lb)
        else:
            alpha, lb=PowerLaw.max_likelihood(y,
                                              lower_bound_range=lower_bound_range[i],
                                              initial_guess=1.2,
                                              upper_bound=upper_bound)
            # must add wrapper for correction to take in a lower bound arg (and disregard it)
            correction_ = powerlaw_correction_spline()
            correction = lambda alpha, K, lb=None: correction_(alpha, K)
            alpha += correction(alpha, (y>=lb).sum())

        # measure the scaling over at least a single order of magnitude
        if (np.log10(y.max())-np.log10(lb))<2:
            alpha1 = np.nan
        else:
            if discrete:
                alpha1=DiscretePowerLaw.mean_scaling(y[y>=lb],
                                         np.logspace(np.log10(y.min())+1., np.log10(y.max()), 5)[1:])
            else:
                alpha1=PowerLaw.mean_scaling(y[y>=lb],
                                         np.logspace(np.log10(y.min())+1., np.log10(y.max()), 5)[1:])
    
        # KS statistics
        if discrete:
            dpl = DiscretePowerLaw(alpha=alpha, lower_bound=lb, upper_bound=upper_bound)
        else:
            dpl = PowerLaw(alpha=alpha, lower_bound=lb, upper_bound=upper_bound)
        ksval=dpl.ksval(y[y>=lb])
        # only calculate p-value if the fit is rather close
        if n_boot_samples>0 and ksval<=ksval_threshold:
            pval, ksSample, (alphaSample,lbSample) = dpl.clauset_test(y[y>=lb], 
                                                                      ksval,
                                                                      lower_bound_range[i], 
                                                                      n_boot_samples,
                                                                      samples_below_cutoff=y[y<lb],
                                                                      return_all=True,
                                                                      correction=correction,
                                                                      n_cpus=1)
        else:
            pval=np.nan
            ksSample=None
            alphaSample=None
            lbSample=None
        print("Done fitting data set %d."%i)
        return alpha, lb, alpha1, ksval, ksSample, pval, alphaSample, lbSample
    
#     for (i,y) in enumerate(Y):
#         f((i,y))
#     return
    pool=Pool(n_cpus)
    alpha, lb, alpha1, ksval, ksSample, pval, alphaSample, lbSample=list(zip(*pool.map(f, enumerate(Y))))
    pool.close()

    alpha=np.array(alpha)
    lb=np.array(lb)
    alpha1=np.array(alpha1)
    ksval=np.array(ksval)
    ksSample=np.array(ksSample)
    pval=np.array(pval)
    alphaSample=np.array(alphaSample)
    lbSample=np.array(lbSample)
    
    if discrete:
        cdfs=[DiscretePowerLaw.cdf(alpha=alpha[i], lower_bound=lb[i]) if not np.isnan(alpha[i]) else None
              for i in range(len(Y))]
    else:
        cdfs=[PowerLaw.cdf(alpha=alpha[i], lower_bound=lb[i]) if not np.isnan(alpha[i]) else None
              for i in range(len(Y))]
    
    fullecdfs=[]
    ecdfs=[]
    for i,d in enumerate(Y):
        fullecdfs.append( ECDF(d) )
        if np.isnan(lb[i]):
            ecdfs.append(None)
        elif (d>=lb[i]).any():
            ecdfs.append( ECDF(d[d>=lb[i]]) )
        else:
            ecdfs.append(None)
    
    return alpha, alpha1, lb, cdfs, ecdfs, fullecdfs, ksval, ksSample, pval, alphaSample, lbSample

def _vtess_loop(args):
    """
    Loop through increasingly finer resolutions. 

    Parameters
    ----------
    args : list
        (p, fname, initial_coarse_grid)
        p : list of denominators to divide pi by. these should be increasing (finer resolution)
        fname : name under which to save results (folder is given by p) and excluding '.p' suffix
        initial_coarse_grid : optional coarse grid to use for first sampling run
    """
    from numpy import pi
    if len(args)==2:
        p, fname=args
        initial_coarse_grid=None
    elif len(args)==3:
        p, fname, initial_coarse_grid=args
    else:
        raise Exception
    
    coarseGrid=initial_coarse_grid
    for p_ in p:
        poissd=PoissonDiscSphere(pi/p_,
                                 coarse_grid=coarseGrid,
                                 width_bds=(3/180*pi, 94/180*pi),
                                 height_bds=(-45/180*pi, 47/180*pi))
        poissd.sample()
        coarseGrid=poissd.samples
    
        pickle.dump({'poissd':poissd}, open('voronoi_grids/%d/%s.p'%(p_, fname), 'wb'), -1)
        print('Done with files %s resolution %d'%(fname,p_))
    
def voronoi(piDenom, nStarts, fileno_start=0, n_jobs=None, initial_coarse_grid=None):
    """Run voronoi tessellation. Results are saved into the voronoi_grids folder. Chain coarse
    gridding such that results of a coarser grid are fed into the finer grid.
    
    Parameters
    ----------
    piDenom : ndarray
        pi is divided by this number to determine the spacing of the Voronoi tessellation
    nStarts : int
    fileno_start : int,0
        Start file name counts from this number.
    """
    import multiprocess as mp
    n_jobs=n_jobs or mp.cpu_count()-1
    assert (np.diff(piDenom)>0).all(), "Grid must get finer."

    # set up directories
    for p in piDenom:
        if not os.path.isdir('voronoi_grids/%d'%p):
            os.mkdir('voronoi_grids/%d'%p)
            
    # setup up thread arguments
    args=[]
    if initial_coarse_grid is None:
        for i in range(nStarts):
            args.append( (piDenom, str(i+fileno_start).zfill(2)) )
    else:
        for i in range(nStarts):
            args.append( (piDenom, str(i+fileno_start).zfill(2), initial_coarse_grid) )

    pool=mp.Pool(n_jobs)
    pool.map(_vtess_loop, args)
    pool.close()

def extract_info_voronoi(subdf, dx, dt, grid_no,
                         this_layer_pixel=None,
                         next_layer_pixel=None,
                         split_geo=True,
                         split_actor=True,
                         split_date=True):
    """
    Parameters
    ----------
    subdf : pandas.Dataframe
        To split.
    dx : int
        Denominator for pi.
    dt : int
        Number of days to threshold.
    grid_no : int
        Saved tessellation to use. These are contained in voronoi_grids/.
    split_geo : bool,True
    split_actor : bool,True
    split_date : bool,True
        
    Returns
    -------
    geoSplit : list
    eventCount
    duration
    fatalities
    diameter
    """

    assert split_geo or split_actor or split_date

    subdf=subdf.loc[:,('EVENT_DATE','LATITUDE','LONGITUDE','FATALITIES','actors')]
     
    # Split by voronoi tiles.
    if split_geo:
        if not next_layer_pixel is None:
            assert len(this_layer_pixel)==len(subdf)
            geoSplit_=[]
    
            # map onto next layer
            for ix in np.unique(next_layer_pixel):
                eventsToAdd=[]
                matchingPixelsInThisLayer=np.where(ix==next_layer_pixel)[0]
                for ixx in matchingPixelsInThisLayer:
                    eventsToAdd.append(subdf.iloc[this_layer_pixel==ixx])

                geoSplit_.append(pd.concat(eventsToAdd).sort_values('EVENT_DATE', axis=0))

        else:
            # do lowest level tiling
            poissd=pickle.load(open('voronoi_grids/%d/%s.p'%(dx, str(grid_no).zfill(2)), 'rb'))['poissd']
            # an offset of 330 degrees center the tessellation above Africa
            geoSplit_, pixIx=pixelate_voronoi(subdf, poissd, 330)
    else:
        geoSplit_=[subdf]
    
    # Split by actors.
    if split_actor:
        geoSplit_=[split_by_actor(s)[0] for s in geoSplit_]
        geoSplit_=list(chain.from_iterable(geoSplit_))
    elif not split_geo:
        geoSplit_=[subdf]

    # Split by date.
    if split_date:
        geoSplit=[]
        for g in geoSplit_:
            g_=tpixelate(g, dt, subdf['EVENT_DATE'].min(), subdf['EVENT_DATE'].max()) 
            geoSplit+=g_
            l1, l2=sum([sum([len(j) for j in i]) for i in g_]), len(g)
            assert l1==l2,(l1,l2)
    
    try:
        geoSplit;
    except NameError:
        geoSplit=geoSplit_
    
    eventCount, duration, fatalities, diameter=duration_and_size(geoSplit)
    try:
        return geoSplit, eventCount, duration, fatalities, diameter, pixIx
    except:
        return geoSplit, eventCount, duration, fatalities, diameter

def extract_info(subdf,dx,dt,
                 add_random_orientation=False,
                 split_geo=True,
                 split_actor=True,
                 split_date=True):
    """
    Parameters
    ----------
    subdf : pandas.Dataframe
        To split.
    dx : int
        Power of 2**n pixels to use. Special value of -1 is no pixelation.
    dt : int
        Number of days to threshold.
    add_random_orientation : bool,False
        Rotate events randomly around the globe (this is for reducing the effect of a 
        particular grid chosen).
    split_geo : bool,True
    split_actor : bool,True
    split_date : bool,True
        
    Returns
    -------
    geoSplit : list
    eventCount
    duration
    fatalities
    diameter
    """
    assert split_geo or split_actor or split_date

    subdf=subdf.loc[:,('EVENT_DATE','LATITUDE','LONGITUDE','FATALITIES','actors')]
    if add_random_orientation:
        randlat,randlon=add_random_orientation
        subdf.loc[:,'LATITUDE']=((subdf.loc[:,'LATITUDE']+randlat)%180)-90
        subdf.loc[:,'LONGITUDE']=((subdf.loc[:,'LONGITUDE']+randlon)%360)-180
    
    # Split by healpy pixels
    if split_geo:
        if dx==-1:
            geoSplit_, pixelDiameter=[subdf], 6370*np.pi
        else:
            geoSplit_, pixelDiameter=pixelate(subdf, 2**dx)
    else:
        geoSplit_=[np.arange(len(subdf), dtype=int)]
    
    # Split by actors.
    if split_actor:
        geoSplit_=[[s[ix] for ix in split_by_actor(subdf.iloc[s.tolist(),:])[0]] for s in geoSplit_]
        geoSplit_=list(chain.from_iterable(geoSplit_))
    elif not split_geo:
        geoSplit_=[np.arange(len(subdf), dtype=int)]

    # Split by date.
    if split_date:
        geoSplit=[]
        for g in geoSplit_:
            g_=tpixelate(subdf.iloc[g], dt, subdf['EVENT_DATE'].min(), subdf['EVENT_DATE'].max()) 
            geoSplit+=[g[i] for i in g_]
            l1, l2=sum([len(i) for i in g_]), len(g)
            assert l1==l2, (l1,l2)
    
    try:
        geoSplit;
    except NameError:
        geoSplit=geoSplit_
    
    eventCount, duration, fatalities, diameter=duration_and_size([subdf.iloc[ix] for ix in geoSplit])
    return geoSplit,eventCount,duration,fatalities,diameter

def setup_quickload(path):
    """Since the pickles contain geoSplit which is a large variable, write quickload files 
    that allow you to avoid loading that. Iterate through all files in the given directory assuming
    that they're pickles.
    
    Parameters
    ----------
    path : str
        For example 'geosplits/battle/full_data_set'
    """
    import os
    
    for f in [i for i in os.listdir(path) if not 'quick' in i]:
        # If quickload file doesn't already exist.
        if not os.path.isfile('%s/%s.quick'%(path,f)):
            indata=pickle.load(open('%s/%s'%(path,f),'rb'))

            del indata['geoSplit']

            pickle.dump(indata,open('%s/%s.quick'%(path,f),'wb'),-1)

