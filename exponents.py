from .acled_utils import *
from data_sets.acled import *
import pickle

TMIN=2
FMIN=2
SMIN=2


def scaling_nu(dr, n_points, nfiles=10, prefix=''):
    """
    Parameters
    ----------
    dr : str
    n_points : list
    nfiles : int,10
    prefix : str,''

    Returns
    -------
    nu : ndarray
    errnu : ndarray
    """

    nu=[] 
    errnu=[]

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        diameter=[d[d>0] for d in indata['diameter']]
        Dmean=[i.mean() for i in diameter]
        Dmax=[i.max() for i in diameter]
        nu.append(np.zeros(len(n_points)))
        errnu.append(np.zeros(len(n_points)))
        
        for i,n in enumerate(n_points):
            soln=2 - loglog_fit(Dmax[-n:], Dmean[-n:], p=2)
            nu[-1][i]=soln[0]
            errnu[-1][i]=loglog_fit_err_bars(Dmax[-n:], Dmean[-n:], soln)[0]

    return np.vstack(nu), np.vstack(errnu)

def _fractal_dimension(x, y, initial_guess=1., rng=(.2,1.5), return_grid=False, return_err=True):
    """Find the fractional dimension df such that <x^df> ~ <y>.

    These averages should be related 1:1 when df is correct.

    Parameters
    ----------
    x : list
        Each element should contain many entries that will be used to calculated the sample mean.
    y : list

    Returns
    -------
    df : float
    errbds : twople
        Error bars on the loglog_fit exponent parameter measured by the standard deviation of the
        log residuals.
    """

    from scipy.optimize import minimize,brute
    ym=[i.mean() for i in y]
    
    # grid sampling is often necessary for convergence
    soln=brute( lambda df:(loglog_fit([(i**df).mean() for i in x], ym)[0] - 1)**2, (rng,),
                Ns=20, full_output=True)
    
    if return_err:
        # error is estimated by looking at fluctuation in exponent parameter in loglog_fit and seeing
        # how that corresponds to fluctuations in the cost function comparing the fractal dimension to
        # 1. then that is mapped to confidence intervals in the fractal dimension
        df=soln[0]
        xm=[(i**df).mean() for i in x]
        logfitParams=loglog_fit(xm, ym)
        logfitErr=loglog_fit_err_bars(xm, ym, logfitParams)  # typically very tight bounds

        # range of variation in cost function
        maxvar=max((logfitErr[0]-1)**2, (logfitErr[1]-1)**2)
        errbds=_fractal_dimension_error(x, ym, df, logfitParams[0],
                                        threshold=maxvar)

    if return_grid:
        return soln[0], errbds, (soln[2], soln[3])
    return soln[0], errbds

def _fractal_dimension_error(x, y, df, params, threshold=.01, eps=1e-6):
    """Find the range of fractal dimension allowed such that the squared error doesn't exceed given
    threshold.

    Parameters
    ----------
    x : list
        Each element should contain many entries that will be used to calculated the sample mean.
    y : list
    df : float
    threshold : twople,.1

    Returns
    -------
    dfBds : (float,float)
    """
    
    from scipy.optimize import minimize_scalar
    
    if type(threshold) is float or type(threshold) is np.float64:
        threshold=(threshold, threshold)
    ym=[i.mean() for i in y]
    
    # no way to get positive error bounds if the estimated parameters are negative
    if params<0:
        return np.nan, np.nan
    
    xm=lambda df, x=x: [(i**df).mean() for i in x] 
    lcost=np.vectorize(lambda df,x=x,ym=ym : abs(abs(loglog_fit(xm(df), ym)[0] - 1) - np.sqrt(threshold[0])) )
    ucost=np.vectorize(lambda df,x=x,ym=ym : abs(abs(loglog_fit(xm(df), ym)[0] - 1) - np.sqrt(threshold[1])) )

    lb=.1
    while lb>1e-3:
        try:
            bds=(minimize_scalar(lcost, bounds=(max([min([lb,df-1]),0]),df-eps), method='bounded')['x'],
                 minimize_scalar(ucost, bounds=(df+eps,df+2), method='bounded')['x'])
            return bds 
        except ValueError:
            lb/=2

    print("Problem calculating fractal dimension error bounds.")
    return (np.nan,np.nan)

def fractal_dimension(dr, n_points, nfiles=10, prefix=''):
    """Find the fractional dimension df such that <x^df> ~ <y>. But cannot calculate temporal
    dimension because z<1 and so mean does not scale with L.

    Get estimates of error bars by looking at where error landscape intersects with given value.

    Parameters
    ----------
    dr : str
    n_points : list
    nfiles : int,10
    prefix : str,''

    Returns
    -------
    ds,df : ndarrays
    """
    
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize

    ds=[]  # size fractal dimension
    df=[]  # fatalities fractal dimension
    eds, edf=[],[]

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        diameter=[d[d>0] for d in indata['diameter']]
        eventCount=[s[s>=SMIN] for s in indata['eventCount']]
        fatalities=[f[f>=FMIN] for f in indata['fatalities']]
        ds.append(np.zeros(len(n_points)))
        eds.append(np.zeros((len(n_points),2)))
        df.append(np.zeros(len(n_points)))
        edf.append(np.zeros((len(n_points),2)))
        
        for i,n in enumerate(n_points):
            df[-1][i], grid=_fractal_dimension(diameter[-n:], fatalities[-n:], return_grid=True,
                                               rng=(1,3))
            # Find width of the values of x such that they coincide with 0.1
            curve=interp1d(*grid, kind='cubic')
            f=lambda x:(curve(x)-.1)**2
            edf[-1][i,0]=minimize(f, 1.1)['x']
            edf[-1][i,1]=minimize(f, 1.9)['x']

            ds[-1][i], grid=_fractal_dimension(diameter[-n:], eventCount[-n:], return_grid=True,
                                               rng=(1,3))
            curve=interp1d(*grid, kind='cubic')
            f=lambda x:(curve(x)-.1)**2
            eds[-1][i,0]=minimize(f, 1.1)['x']
            eds[-1][i,1]=minimize(f, 1.9)['x']

    return np.vstack(ds), np.vstack(df), eds, edf

def df_scaling(dr, n_points, nfiles=10, prefix=''):
    """Find the fractional dimension df such that <x^df> ~ <y>. But cannot calculate temporal
    dimension because z<1 and so mean does not scale with L.

    Get estimates of error bars by looking at where error landscape intersects with given value.

    Parameters
    ----------
    dr : str
    n_points : list
    nfiles : int,10
    prefix : str,''

    Returns
    -------
    df : ndarrays
        Estimated fractal dimension. Each row is a different file.
    edf : list
        Each element is the upper and lower bounds for each n_point in a single file.
    """
    
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import minimize

    df=[]  # fatalities fractal dimension
    edf=[]

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        diameter=[d[d>0] for d in indata['diameter']]
        eventCount=[s[s>=SMIN] for s in indata['eventCount']]
        fatalities=[f[f>=FMIN] for f in indata['fatalities']]
        df.append(np.zeros(len(n_points)))
        edf.append(np.zeros((len(n_points),2)))
        
        for i,n in enumerate(n_points):
            df[-1][i], grid=_fractal_dimension(diameter[-n:], fatalities[-n:], return_grid=True,
                                               rng=(.5,3))
            # Find width of the values of x such that they coincide with the error increasing by 0.1
            curve=UnivariateSpline(*grid, ext='const', s=1e-5)
            f=lambda x:(curve(x)-.1)**2
            edf[-1][i,0]=minimize(f, df[-1][i]-.1, method='nelder-mead')['x'][0]
            edf[-1][i,1]=minimize(f, df[-1][i]+.1, method='nelder-mead')['x'][0]

    return np.vstack(df), edf

def distributions(dr, nfiles=10, prefix=''):
    """Measure exponents for P(S), P(F), P(T), P(L) from the distributions with max likelihood. Returned
    errors represent a factor of exp(-.5) drop per data point. This is akin to the drop in likelihood for a
    Gaussian when we go away by a single standard deviation.
    
    Parameters
    ----------
    dr : str
    nfiles : int,10
    prefix : str,''
        File prefixes for loading.
        
    Returns
    -------
    nu,upsilon,tau,alpha, enu, eupsilon, etau, ealpha
    """
    
    from misc.stats import PowerLaw
    from misc.stats import DiscretePowerLaw as dpl

    nu=[]       # diameter of events
    tau=[]      # sizes
    upsilon=[]  # fatalities
    alpha=[]    # duration
    enu, etau, eupsilon, ealpha=[],[],[],[]

    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        diameter=indata['diameter']
        eventCount=indata['eventCount']
        fatalities=indata['fatalities']
        duration=indata['duration']
        dayThreshold=indata['dayThreshold']
        n=len(dayThreshold)
        nu.append(np.zeros(n))
        tau.append(np.zeros(n))
        upsilon.append(np.zeros(n))
        alpha.append(np.zeros(n))
        enu.append(np.zeros((2,n)))
        etau.append(np.zeros((2,n)))
        eupsilon.append(np.zeros((2,n)))
        ealpha.append(np.zeros((2,n)))

        # Event length (nu)
        for i,d in enumerate(diameter):
            lowerBound=100
            keepix=d>=lowerBound
            if keepix.sum():
                nu[fileix][i]=PowerLaw.max_likelihood_alpha(d[keepix], lower_bound=lowerBound)
                enu[fileix][:,i]=PowerLaw.alpha_range(d[keepix], nu[fileix][i], .5*keepix.sum(),
                                                      lower_bound=lowerBound)

        # Event size (tau)
        for i,n in enumerate(eventCount):
            keepix=n>=SMIN
            _, soln=dpl.max_likelihood_alpha(n[keepix], lower_bound=SMIN, full_output=True)
            if soln['success'] or 'precision' in soln['message']:
                tau[fileix][i]=soln['x']
                etau[fileix][:,i]=dpl.alpha_range(n[keepix], tau[fileix][i], .5*keepix.sum(),
                                                  lower_bound=SMIN)
            else:
                tau[fileix][i]=nan

        # Fatalities (upsilon)
        for i,f in enumerate(fatalities):
            keepix=fatalities[i]>=FMIN
            _,soln=dpl.max_likelihood_alpha(f[keepix], lower_bound=FMIN, initial_guess=1.6, full_output=True)
            if soln['success'] or 'precision' in soln['message']:
                upsilon[fileix][i]=soln['x']
                eupsilon[fileix][:,i]=dpl.alpha_range(f[keepix], upsilon[fileix][i], .5*keepix.sum(),
                                                      lower_bound=FMIN)
            else:
                upsilon[fileix][i]=nan

        # Event duration.
        # Assert lower bound as the corresponding dayThreshold+1.
        for i,n in enumerate(duration):
            keepix=n>=(dayThreshold[i]*2)
            _,soln=dpl.max_likelihood_alpha(n[keepix],
                                            lower_bound=dayThreshold[i]*2,
                                            upper_bound=np.ceil(7304/dayThreshold[i])*dayThreshold[i],
                                            full_output=True)
            if soln['success'] or 'precision' in soln['message']:
                alpha[fileix][i]=soln['x']
                # Error bars represent 4 std deviations assuming Gaussian distribution
                ealpha[fileix][:,i]=dpl.alpha_range(n[keepix], alpha[fileix][i], .5*keepix.sum(),
                                                    lower_bound=dayThreshold[i]*2, 
                                                    upper_bound=np.ceil(7304/dayThreshold[i])*dayThreshold[i])
            else:
                alpha[fileix][i]=nan
    
    nu=np.vstack(nu)
    tau=np.vstack(tau)
    upsilon=np.vstack(upsilon)
    alpha=np.vstack(alpha)

    # Calculate relationships for critical exponents.
    #betaSizes=(1-alpha)/(1-tau)
    #betaFatalities=(1-alpha)/(1-upsilon)
    
    return nu, upsilon, tau, alpha, enu, eupsilon, etau, ealpha

def exponent_relation_1(x,y,iprint=True):
    """Check that the mean scales with the cutoff for the distribution. For example, that
    <F> ~ F_c^{2-tau}
    
    Parameters
    ----------
    x : avalanche distribution
    y : scaling of mean given max
    iprint : bool,True

    Returns
    -------
    x,2-y
    """
    if iprint:
        print('Max lik.\tScaling')
        print('%1.2f\t\t%1.2f'%(x,2-y))
        print()
    return (x,2-y)
    
def exponent_relation_2(x,y,z,iprint=True):
    """
    Parameters
    ----------
    x : float
        Fractal dimension
    y : float
        Distribution of lengths.
    z : float
        distribution of sizes.
    iprint : bool,True

    Returns
    -------
    zfromxy : float
    yfromxz : float
    zfromxy : float
    """
    if iprint:
        print("Leaving out x")
        print('%1.2f\t%1.2f'%(x,(1-y)/(1-z)))
        print("Leaving out y")
        print('%1.2f\t%1.2f'%(y,1-x*(1-z)))
        print("Leaving out z")
        print('%1.2f\t%1.2f'%(z,1-(1-y)/x))
        print()
    
    return (x,(1-y)/(1-z)),(y,1-x*(1-z)),(z,1-(1-y)/x)

def raw_scaling_with_L(dr, nfiles=10, prefix=''):
    """Compare each event diameter with the number of events that happened. This will give the
    uncorrected fractal scaling exponents (corrected exponent scales with the pixel size).

    Parameters
    ----------
    dr : str
    nfiles : int,10
    prefix : str,''

    Returns
    -------
    uz, uds, udf : ndarrays
        Uncorrected exponents.
    """

    from .acled_utils import loglog_fit,loglog_fit_err_bars

    uz=[]   # dynamic critical exponent
    uds=[]  # size fractal dimension
    udf=[]  # fatalities fractal dimension
    errz=[]
    errds=[]
    residz=[]
    residds=[]  # hessian for exponent estimation

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        duration=indata['duration']
        diameter=indata['diameter']
        eventCount=indata['eventCount']
        fatalities=indata['fatalities']
        pixDiameter=indata['pixDiameter']
        uz.append(np.zeros(len(pixDiameter)))
        uds.append(np.zeros(len(pixDiameter)))
        udf.append(np.zeros(len(pixDiameter)))
        errz.append(np.zeros(len(pixDiameter)))
        errds.append(np.zeros(len(pixDiameter)))
        residz.append([])
        residds.append([])
        
        for i,(dur,diam) in enumerate(zip(duration, diameter)):
            keepix=(dur>=TMIN)&(diam>0)
            soln=loglog_fit(diam[keepix], dur[keepix])
            uz[fileix][i]=soln[0]

            fitErr=loglog_fit_err_bars(diam[keepix], dur[keepix], soln)
            errz[-1][i]=fitErr[0]
            residz[-1].append(((np.log(dur[keepix]) - soln[0]*np.log(diam[keepix])-soln[1]),
                               (np.log(dur[keepix])/soln[0] - np.log(diam[keepix])-soln[1]/soln[0])))

        for i,(s,diam) in enumerate(zip(eventCount, diameter)):
            keepix=(diam>0)&(s>=SMIN)
            soln=loglog_fit(diam[keepix], s[keepix], p=1)
            uds[fileix][i]=soln[0]

            fitErr=loglog_fit_err_bars(diam[keepix], s[keepix], soln)
            errds[-1][i]=fitErr[0]
            residds[-1].append(((np.log(s[keepix]) - soln[0]*np.log(diam[keepix])-soln[1]),
                                (np.log(s[keepix])/soln[0] - np.log(diam[keepix])-soln[1]/soln[0])))

        for i,(f,diam) in enumerate(zip(fatalities, diameter)):
            keepix=(diam>0)&(f>=FMIN)
            udf[fileix][i]=loglog_fit(diam[keepix], f[keepix])[0]

    return np.vstack(uz), np.vstack(uds), np.vstack(udf), errz, residz, errds, residds

def scaling_with_L(dr, n_points, corrected=True, nfiles=10, prefix=''):
    """Mean scaling relation with pixel diameter to estimate fractal dimensions. These are not the
    straightforward fractal dimensions because an integral must be done to calculate the mean.
        
    Parameters
    ----------
    dr : str
    n_points : list
        Number of points from the last point to use to estimate the log slope.
    corrected : bool,True
        If True, return scaling with pixel diameter. Else return scaling with mean of event extent.
    nfiles : int,10
    prefix : str,''
    
    Returns
    -------
    z,dl,ds,df
    """
    
    from .acled_utils import loglog_fit
    n=len(n_points)
    z=np.zeros((nfiles, n))   # dynamic critical exponent
    dl=np.zeros((nfiles, n))  # extent fractal dimension
    ds=np.zeros((nfiles, n))  # size fractal dimension
    df=np.zeros((nfiles, n))  # fatalities fractal dimension
    
    if corrected:
        # Diameter here is the pixel diameter
        for fileix in range(nfiles):
            indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
            lmax=[i.max() for i in indata['diameter']]
            tmean=[i[i>0].mean() for i in indata['duration']]
            smean=[i[i>1].mean() for i in indata['eventCount']]
            fmean=[i[i>1].mean() for i in indata['fatalities']]
            pixDiameter=indata['pixDiameter']
            
            for i,npts in enumerate(n_points):
                z[fileix,i]=loglog_fit(pixDiameter[-npts:], tmean[-npts:])[0]
                dl[fileix,i]=loglog_fit(lmax[-npts:], pixDiameter[-npts:])[0]
                ds[fileix,i]=loglog_fit(pixDiameter[-npts:], smean[-npts:])[0]
                df[fileix,i]=loglog_fit(pixDiameter[-npts:], fmean[-npts:])[0]
        return z, dl, ds, df

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        lmax=[i.max() for i in indata['diameter']]
        lmean=[i[i>0].mean() for i in indata['diameter']]
        tmean=[i[i>0].mean() for i in indata['duration']]
        smean=[i[i>1].mean() for i in indata['eventCount']]
        fmean=[i[i>1].mean() for i in indata['fatalities']]
        
        for i,npts in enumerate(n_points):
            z[fileix,i]=loglog_fit(lmean[-npts:], tmean[-npts:])[0]
            dl[fileix,i]=loglog_fit(lmean[-npts:], lmax[-npts:])[0]
            ds[fileix,i]=loglog_fit(lmean[-npts:], smean[-npts:])[0]
            df[fileix,i]=loglog_fit(lmean[-npts:], fmean[-npts:])[0]
    return z, dl, ds, df

def scaling_with_T(dr, n_points, nfiles=10, prefix=''):
    """Mean scaling relation between means. Alternative way of measuring ratios of scaling
    exponents.
        
    Parameters
    ----------
    dr : str
    n_points : list
        Number of points from the last point to use to estimate the log slope.
    nfiles : int,10
    prefix : str,''
    
    Returns
    -------
    dsz,dfz
    """
    
    from .acled_utils import loglog_fit
    n=len(n_points)
    dsz=np.zeros((nfiles, n))  # ds/z
    dfz=np.zeros((nfiles, n))  # df/z
    
    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        tmean=[i[i>0].mean() for i in indata['duration']]
        smean=[i[i>1].mean() for i in indata['eventCount']]
        fmean=[i[i>1].mean() for i in indata['fatalities']]
        pixDiameter=indata['pixDiameter']
        
        for i,npts in enumerate(n_points):
            dsz[fileix,i]=loglog_fit(tmean[-npts:], smean[-npts:])[0]
            dfz[fileix,i]=loglog_fit(tmean[-npts:], fmean[-npts:])[0]

    return dsz, dfz

def scaling_F_by_T(dr, nfiles=10, prefix=''):
    """Scaling relation between all fatality data points and durations t.
        
    Parameters
    ----------
    dr : str
    n_points : list
        Number of points from the last point to use to estimate the log slope.
    nfiles : int,10
    prefix : str,''
    
    Returns
    -------
    """
    
    dfz=[]
    
    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        fatalities=indata['fatalities']
        durations=indata['duration'] 
        
        dfz.append(np.zeros(len(durations)))
        for i,(x,y) in enumerate(zip(durations, fatalities)):
            keepix=(x>0)&(y>1)
            dfz[-1][i]=loglog_fit(x[keepix], y[keepix])[0]
    return np.vstack(dfz)

def scaling_S_by_F(dr, n_points, nfiles=10, prefix=''):
    """Mean scaling relation between means <S> and <F>.
    exponents.
        
    Parameters
    ----------
    dr : str
    n_points : list
        Number of points from the last point to use to estimate the log slope.
    nfiles : int,10
    prefix : str,''
    
    Returns
    -------
    """
    
    n=len(n_points)
    x=np.zeros((nfiles, n)) 
    y=np.zeros((nfiles, n)) 
    z=np.zeros((nfiles, n)) 
    
    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr, prefix, str(fileix).zfill(3)), 'rb'))
        smean=[i[i>1].mean() for i in indata['eventCount']]
        fmean=[i[i>1].mean() for i in indata['fatalities']]
        lmax=[i.max() for i in indata['diameter']]
        
        for i,npts in enumerate(n_points):
            x[fileix,i]=loglog_fit(fmean[-npts:], smean[-npts:])[0]
            y[fileix,i]=loglog_fit(lmax[-npts:], smean[-npts:])[0]
            z[fileix,i]=loglog_fit(lmax[-npts:], fmean[-npts:])[0]

    return x, y, z

def scaling_relations(dr, nSample, pixDiameter, nfiles=10, prefix=''):
    """measure exponents by using the scaling of the mean and max with pixel diameter
    
    Parameters
    ----------
    dr : str
    nSample : int
    pixDiameter : 
    nfiles : int,10
    prefix : str,''
    
    Returns
    -------
    """
    from .acled_utils import loglog_fit
    La=np.zeros(nfiles)
    Lb=np.zeros(nfiles)
    Lc=np.zeros(nfiles)
    Ta=np.zeros(nfiles)  # duration mean scaling with distance
    Tb=np.zeros(nfiles)  # duration max scaling with distance
    Tc=np.zeros(nfiles)
    Sa=np.zeros(nfiles)
    Sc=np.zeros(nfiles)
    Sb=np.zeros(nfiles)
    Fa=np.zeros(nfiles)
    Fb=np.zeros(nfiles)
    Fc=np.zeros(nfiles)
    dsScaling=np.zeros(nfiles)
    zScaling=np.zeros(nfiles)
    dfScaling=np.zeros(nfiles)
    betaScalingSizes=np.zeros(nfiles)
    betaScalingFatalities=np.zeros(nfiles)

    # Diameter here is the data diameter
    for fileix in range(nfiles):
        indata=pickle.load(open('geosplits/%s/%s%s.p.quick'%(dr,prefix,str(fileix).zfill(3)),'rb'))
        diameter=indata['diameter']
        duration=indata['duration']
        eventCount=indata['eventCount']
        fatalities=indata['fatalities']

        Lmean=[np.nanmean(i[i>0]) for i in diameter][-nSample:]
        Lmax=[np.nanmax(i[i>0]) for i in diameter][-nSample:]
        La[fileix]=loglog_fit(pixDiameter[-nSample:],Lmean)[0]
        Lb[fileix]=loglog_fit(pixDiameter[-nSample:],Lmax)[0]
        Lc[fileix]=loglog_fit(Lmean,Lmax)[0]

        Tmean=[np.nanmean(i[i>0]) for i in duration][-nSample:]
        Tmax=[np.nanmax(i[i>0]) for i in duration][-nSample:]
        Ta[fileix]=loglog_fit(pixDiameter[-nSample:],Tmean)[0]
        Tb[fileix]=loglog_fit(pixDiameter[-nSample:],Tmax)[0]
        Tc[fileix]=loglog_fit(Tmean,Tmax)[0]

        Smean=[np.nanmean(i[i>1]) for i in eventCount[-nSample:]]
        Smax=[np.nanmax(i[i>1]) for i in eventCount[-nSample:]]
        Sa[fileix]=loglog_fit(pixDiameter[-nSample:],Smean)[0]
        Sb[fileix]=loglog_fit(pixDiameter[-nSample:],Smax)[0]
        Sc[fileix]=loglog_fit(Smean,Smax)[0]

        Fmean=[np.nanmean(i[i>1]) for i in fatalities][-nSample:]
        Fmax=[np.nanmax(i[i>1]) for i in fatalities][-nSample:]
        Fa[fileix]=loglog_fit(pixDiameter[-nSample:],Fmean)[0]
        Fb[fileix]=loglog_fit(pixDiameter[-nSample:],Fmax)[0]
        Fc[fileix]=loglog_fit(Fmean,Fmax)[0]
        
        dsScaling[fileix]=loglog_fit(Lmean, Smean)[0]
        zScaling[fileix]=loglog_fit(Lmean, Tmean)[0]
        dfScaling[fileix]=loglog_fit(Lmean, Fmean)[0]
        betaScalingSizes[fileix]=loglog_fit(Smean,Tmean)[0]
        betaScalingFatalities[fileix]=loglog_fit(Fmean,Tmean)[0]
        
    return Fc, Sc, Lc, dsScaling, z, dfScaling, betaScalingSizes, betaScalingFatalities

def exponent_comparison(dr, df, ds, z, nu, alpha, upsilon, tau, pixDiameter, nSampleRange,
                        nfiles=10, prefix=''):
    import matplotlib.pyplot as plt
    from .acled_utils import exponent_relation_1,exponent_relation_2
    assert all([i>1 for i in nSampleRange])
    
    fig, ax=plt.subplots(figsize=(20,3), ncols=5)
    
    for i,nSample in enumerate(nSampleRange):
        out=scaling_relations(dr,
                              nSample,
                              pixDiameter,
                              nfiles,
                              prefix=prefix)
        Fc, Sc, Lc, dsScaling, z, dfScaling, betaScalingSizes, betaScalingFatalities=out

        mdf, errdf=df[:,-nSample:].mean(), df[:,-nSample:].std()
        mds, errds=ds[:,-nSample:].mean(), ds[:,-nSample:].std()
        mz, errz=z[:,-nSample:].mean(), z[:,-nSample:].std()
        mNu, errNu=nu[:,-nSample:].mean(), nu[:,-nSample:].std()
        mAlpha, errAlpha=alpha[:,-nSample:].mean(), alpha[:,-nSample:].std()
        mUpsilon, errUpsilon=upsilon[:,-nSample:].mean(), upsilon[:,-nSample:].mean()
        mTau, errTau=tau[:,-nSample:].mean(), tau[:,-nSample:].std()

        # Fatalities
        # Scaling between mean and cutoff
        y=exponent_relation_1(mUpsilon, 1/Fc.mean(), False)
        ax[0].plot(nSample, y[0], 'bo')
        ax[0].plot(nSample, y[1], 'rx')
        
        y=exponent_relation_1(mNu, 1/Lc.mean(), False)
        ax[1].plot(nSample, y[0], 'bo')
        ax[1].plot(nSample, y[1], 'rx')

        # making this consistent makes the prediction for nu=1.9
        y=exponent_relation_2(mdf, 1.91, mUpsilon, False)
        ax[1].plot(nSample, y[1][1], 'g^')
        
#         # Scaling between mean size and cutoff
#         y=exponent_relation_1(mTau,1/Sc.mean(),False)
#         ax[0].plot(nSample,y[0],'bo')
#         ax[0].plot(nSample,y[1],'rx')

        # Durations
        ax[2].plot(nSample,mAlpha,'bo')

        ax[3].plot(nSample,betaScalingSizes.mean(),'bo')
        ax[3].plot(nSample,mds/mz,'rx')
        ax[3].plot(nSample,(1-mAlpha)/(1-mTau),'g^')
        
        # Ratio of max-likelihood estimates is far off
        ax[4].plot(nSample,betaScalingFatalities.mean(),'bo')
        ax[4].plot(nSample,mdf/mz,'rx')
        ax[4].plot(nSample,(1-mAlpha)/(1-mUpsilon),'g^')
        
    ylim=[a.get_ylim() for a in ax]
        
    for i,nSample in enumerate(nSampleRange):
        out=scaling_relations(dr,
                              nSample,
                              pixDiameter,
                              nfiles,
                              prefix=prefix)
        Fc, Sc, Lc, dsScaling, z, dfScaling, betaScalingSizes, betaScalingFatalities=out

        mdf,errdf=df[:,-nSample:].mean(),df[:,-nSample:].std()
        mds,errds=ds[:,-nSample:].mean(),ds[:,-nSample:].std()
        mz,errz=z[:,-nSample:].mean(),z[:,-nSample:].std()
        mNu,errNu=nu[:,-nSample:].mean(),nu[:,-nSample:].std()
        mAlpha,errAlpha=alpha[:,-nSample:].mean(),alpha[:,-nSample:].std()
        mUpsilon,errUpsilon=upsilon[:,-nSample:].mean(),upsilon[:,-nSample:].mean()
        mTau,errTau=tau[:,-nSample:].mean(),tau[:,-nSample:].std()
        
        
        # Fatalities
        # Scaling between mean and cutoff
        _,upsScaling=exponent_relation_1(mUpsilon,1/Fc.mean(),False)
        ax[0].errorbar(nSample,mUpsilon,errUpsilon,fmt='bo')
        ax[0].errorbar(nSample+.05,upsScaling,(1/Fc).std(ddof=1),fmt='rx')
        if i==0:
            ax[0].legend(('lik','sc'),fontsize='xx-small')
        
        _,nuScaling1=exponent_relation_1(mNu,1/Lc.mean(),False)
        ax[1].errorbar(nSample,mNu,errNu,fmt='bo')
        ax[1].errorbar(nSample+.05,nuScaling1,(1/Lc).std(ddof=1),fmt='rx')

        # making this consistent makes the prediction for nu=1.9
        _,nuScaling2,_=exponent_relation_2(mdf,1.91,mUpsilon,False)
        nuScaling2=nuScaling2[1]
        ax[1].errorbar(nSample+.1,nuScaling2,
                       (errdf/mdf+errUpsilon/(1-mUpsilon))*mdf*(1-mUpsilon),
                       fmt='g^')
        if i==0:
            ax[1].legend(('lik','sc1','sc2'),fontsize='xx-small')

        _,tauScaling=exponent_relation_1(mTau,1/Sc.mean(),False)

        # Durations
        # Scaling between mean and cutoff should be weak for durations
#         y=exponent_relation_1(mAlpha,1/Tc.mean(),False)
        ax[2].errorbar(nSample,mAlpha,errAlpha,fmt='bo')
#         ax[2].errorbar(nSample,y[1],'rx')
        if i==0:
            ax[2].legend(('lik',),fontsize='xx-small')

        ax[3].errorbar(nSample, betaScalingSizes.mean(), betaScalingSizes.std(ddof=1), fmt='bo')
        ax[3].errorbar(nSample+.05,mds/mz,(errds/mds+errz/mz)*mds/mz,fmt='rx')
        ax[3].errorbar(nSample+.1,(1-mAlpha)/(1-mTau),
                       (errAlpha/(1-mAlpha)+errTau/(1-mTau))*(1-mAlpha)/(1-mTau),
                       fmt='g^')
        if i==0:
            ax[3].legend(('sc1','sc2','lik2'),fontsize='xx-small')
        
        # Ratio of max-likelihood estimates is far off
        ax[4].errorbar(nSample, betaScalingFatalities.mean(), betaScalingFatalities.std(ddof=1), 
                       fmt='bo')
        ax[4].errorbar(nSample+.05, mdf/mz, (errdf/mdf+errz/mz)*mdf/mz, fmt='rx')
        y=(1-mAlpha)/(1-mUpsilon)
        ax[4].errorbar(nSample+.1, y, (errAlpha/(1-mAlpha)+errUpsilon/(1-mUpsilon))*y, fmt='g^')
        if i==0:
            ax[4].legend(('lik1', 'like2', 'lik3'), fontsize='xx-small')
        
    ax[0].set(xlabel='n rescalings', ylabel=r'$\upsilon$')
    ax[1].set(xlabel='n rescalings', ylabel=r'$\nu$')
    ax[2].set(xlabel='n rescalings', ylabel=r'$\alpha$')
    ax[3].set(xlabel='n rescalings', ylabel=r'$\beta_{\rm s}$')
    ax[4].set(xlabel='n rescalings', ylabel=r'$\beta_{\rm f}$')
    for ylim_,ax_ in zip(ylim,ax):
        ax_.set(ylim=ylim_)
    
    fig.subplots_adjust(wspace=.4)
    
    return ( upsScaling,
             tauScaling,
             nuScaling1,
             nuScaling2,
             betaScalingSizes,
             betaScalingFatalities )

