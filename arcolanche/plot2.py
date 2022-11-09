# ====================================================================================== #
# Plotting module for transfer entropy paper.
# Authors: Eddie Lee, edlee@csh.ac.at
#          Niraj Kushwaha, kushwaha@csh.ac.at
# ====================================================================================== #
import numpy as np
import matplotlib.pyplot as plt
from misc.stats import ECDF



def ecdf(ax, X, lb, discrete=True, c='C0', **plot_kw):
    uX = np.unique(X)
    
    if discrete:
        ecdf, ecdflo, ecdfhi = ECDF(X, conf_interval=(2.5,97.5), as_delta=True, n_boot_samples=100)
        ax.errorbar(uX[uX<lb], 1-ecdf(uX[uX<lb]), 
                        yerr=np.vstack((ecdflo(uX[uX<lb]), ecdfhi(uX[uX<lb]))),
                        elinewidth=.5, c='gray',
                        fmt='.', zorder=1)
        ax.errorbar(uX[uX>=lb], 1-ecdf(uX[uX>=lb]), 
                    yerr=np.vstack((ecdflo(uX[uX>=lb]), ecdfhi(uX[uX>=lb]))),
                    elinewidth=.5, c=c,
                    fmt='.', zorder=1)
    else:
        ecdf, ecdflo, ecdfhi = ECDF(X, conf_interval=(2.5,97.5), as_delta=False, n_boot_samples=100)
        
        xplot = np.logspace(np.log10(uX[0]), np.log10(uX[-1]), 300)
        ax.fill_between(xplot[xplot<lb], 1-ecdflo(xplot[xplot<lb]), 1-ecdfhi(xplot[xplot<lb]),
                        facecolor=c,
                        alpha=.5,
                        zorder=1)
        ax.fill_between(xplot[xplot>=lb], 1-ecdflo(xplot[xplot>=lb]), 1-ecdfhi(xplot[xplot>=lb]),
                        facecolor=c,
                        alpha=.5,
                        zorder=1)
        
        ax.plot(xplot[xplot<lb], 1-ecdf(xplot[xplot<lb]), c='gray')
        ax.plot(xplot[xplot>=lb], 1-ecdf(xplot[xplot>=lb]),
                c=c,
                zorder=1)

    x = np.array([lb, uX.max()+1])

    ax.set(xscale='log', yscale='log', **plot_kw)
    
def scaling(ax, X, Y, Xmn, Ymn, c='C0', **plot_kw):
    ix = (X>=Xmn) & (Y>=Ymn)
    ax.loglog(X[~ix], Y[~ix], '.', color='gray')
    ax.loglog(X[ix], Y[ix], '.', c=c)
    ax.set(**plot_kw)
