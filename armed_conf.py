# Armed conflict data set ipynb function.
# 2013-12-11
from numpy import *

def load_data():
    import csv
    with open("../data/ucdp.prio.armed.conflict.v4.2013.csv") as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    print(data[0])

    return data

def get_col(data,colname):
    colix = data[0].index(colname)
    col = [i[colix] for i in data[1:]]
    return col

def fit_data(data,ax):
    """
    Automate fitting.
    2013-12-11
    """
    import distributionfit as df
    import likelihood_fits_fcns as lf
    from misc_fcns import get_ax
    from scipy.special import expi
    from scipy.integrate import quad

    fit = df.Distribution(data,log=True)
    fit.ex = fit.get_exponential(data.min(),data.max())
    fit.ex.disc_pdf()
    print( "dpdf integrates to "+str(sum(fit.ex.dpdf)) )
    l,dl = lf.exp_l_dl(data/1e3)

    # Plot empirical ccdf.
    fit.plot_ccdf(ax)
    fit.calc_ccdf_err()
    ax.plot(fit.x,cumsum(fit.ex.dpdf[::-1])[::-1])
    err = std(fit.cdf['ccdferr'],0, ddof=1)
    mi = fit.cdf['ccdf'] - std(fit.cdf['ccdferr'], 0)
    mi[mi<0] = 1e-30
    mx = fit.cdf['ccdf'] + std(fit.cdf['ccdferr'], 0)
    ix = fit.cdf['ccdf']!=0
    ax.fill_between( fit.cdf['x'][ix],mi[ix],mx[ix],alpha=.25 )

    # Plot fits.
    x = fit.x/1e3
    # Nonstationary poisson distirbution over uniform interval that needs to be normalized
    # over an integral on t. Computation seems to be difficult for relatively large values
    # of delta lambda.
    f = lambda t,l,dl: exp(-l*t)/t* ((l+1./t)*sinh(dl*t) -dl*cosh(dl*t))
    z = quad( lambda t: f(t,l,dl), x[0],inf )[0]
    fnorm = lambda t: f(t,l,dl)/z

    g = vectorize( lambda t1: 1-quad( lambda t: fnorm(t),x[0],t1 )[0] )
    print "l,dl = "+str(l)+","+str(dl)
    ax.plot( fit.x,g(fit.x/1e3) )

    ax.set_ylim([1e-3,1])
    return

def get_dt(dates):
    """
    2013-12-11
        Given list of datetime instances, calcualte the dt between uniquely consecutive
        elements
    """
    _date = dates[0]
    dt = []
    udates = unique(dates)

    for date in udates[1:]:
        _dt = (date-_date).total_seconds()
        for i in range(dates.count(date)):
            dt.append( _dt )
        _date = date
    return array(dt)
