# ====================================================================================== #
# Module for analyzing ACLED data.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

from .utils import *


def africa(fig=None, ax=None):
    """Plot map of Africa.
    """
    
    if not fig is None:
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)

    ax.set_extent(default_extent())

    return ax

def default_extent(degrees=True):
    if degrees:
        return np.array([0.05235987755982988+330/180*np.pi, 1.6406094968746698+330/180*np.pi,
                         -0.5853981633974483, 0.6203047484373349])*180/np.pi
    return np.array([0.05235987755982988+330/180*np.pi, 1.6406094968746698+330/180*np.pi,
                     -0.5853981633974483, 0.6203047484373349])

def cdf(Y, ax, discrete=True, mn=1):
    """Shortcut for plotting CDF.

    Parameters
    ----------
    Y : ndarray
        Sample to plot.
    ax : matplotlib.Axes
    
    Returns
    -------
    None
    """
    
    if discrete:
        p = np.bincount(Y)[mn:]
        p = p / p.sum()
        x = np.arange(mn, p.size+mn)
        
        # last point is, by defn, zero
        ax.loglog(x[:-1], 1 - np.cumsum(p)[:-1], '.')
    else:
        ecdf = ECDF(Y)

        ax.loglog(ecdf.x, 1-ecdf.y, '-')

def setup_gif(t, lonlat, dr='gif', overwrite=False):
    """Plot evolution of conflict avalanche simply by the order in which they appear in
    the data by day.

    Parameters
    ----------
    t : ndarray
    lonlat : ndarray
        Lonlat coordinates.
    dr : str, 'gif'
    overwrite : bool, False
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # check to see if folder exists
    if os.path.isdir(dr):
        if len(os.listdir(dr)) and not overwrite:
            raise Exception
        else:
            for f in os.listdir(dr):
                os.remove('%s/%s'%(dr,f))
    else:
        os.makedirs(dr)
    
    # make plot
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.BORDERS)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_extent([-19, 53, -37, 39], crs=ccrs.PlateCarree())
    textvar = fig.text(0, 0, 'Day 0')
    
    # iterate through all days from 0 til the end and save a figure for each day
    for i in range(t.max()+1):
        ix = i==t
        if ix.any():
            ax.plot(lonlat[ix,0], lonlat[ix,1], '.',
                    c='C1',
                    alpha=.1,
                    mew=0,
                    transform=ccrs.PlateCarree())
        textvar.remove()
        textvar = fig.text(.18, .2, 'Day %d'%i, fontsize=30)
        fig.savefig('%s/%s.png'%(dr,str(i).zfill(5)), dpi=120)

def get_slopes(percent, traj):
    """Slope of temporal profile measured at beginning and end.

    Parameters
    ----------
    percent : float
    traj : ndarray
    """

    assert 1<percent<50
    t = np.linspace(0,1,250)
    ix = np.argmin(np.abs(t-percent/100))
    startSlope = (traj[:,ix]-traj[:,0])/t[ix]
    
    ix = np.argmin(np.abs(t-(1-percent/100)))
    endSlope = (1-traj[:,ix])/(1-t[ix])
    return startSlope, endSlope
