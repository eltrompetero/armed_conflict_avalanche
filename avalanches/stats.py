# ====================================================================================== #
# Statistics between Voronoi cell activity.
# This was written for binary coarse-graining of activity for a local maxent formulation.
# 
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from ..utils import *



# ========= #
# Functions #
# ========= #
def joint_prob(eventsx, eventsy, T):
    """Joint probabilities between events listed in x and y DataFrames. Only considering
    whether any event occurred at all or not. Events are only counted if they occurred
    within adjacent time periods.
    
    Parameters
    ----------
    eventsx : DataFrame
    eventsy : DataFrame
    T : int
        Length of time series.
    
    Returns
    -------
    ndarray
        Probability for variable pairs (x_{t-1}, y_t), (x_t, y_t), (x_t, y_{t-1}).
        Each row considers all possible pairs of activation in order of (00, 01, 10, 11).
    """
    
    tx = np.zeros(T+1, dtype=int)
    ty = np.zeros(T+1, dtype=int)
    
    if not len(eventsx)==0:
        ix = eventsx.tpixel
        tx[ix] = 1
    if not len(eventsy)==0:
        iy = eventsy.tpixel
        ty[iy] = 1
    
    # joint probability matrix
    p = np.zeros((3,4))
    p[0] = _pair_probs(tx[:-1], ty[1:])
    p[1] = _pair_probs(tx, ty)
    p[2] = _pair_probs(tx[1:], ty[:-1])

    return p

def self_joint_prob(eventsx, T):
    """Self joint probabilities between events listed in x DataFrame. Only considering
    whether any event occurred at all or not. Events are only counted if they occurred
    within adjacent time periods.
    
    Parameters
    ----------
    eventsx : DataFrame
    T : int
        Length of time series.
    
    Returns
    -------
    ndarray
        Probability for variable pairs (x_{t-1}, x_t). All possible pairs of activation in
        order of (00, 01, 10, 11).
    """
    
    tx = np.zeros(T+1, dtype=int)
    
    if not len(eventsx)==0:
        ix = eventsx.tpixel
        tx[ix] = 1
    
    # joint probability matrix
    p = np.zeros((1,4))
    p[0] = _pair_probs(tx[:-1], tx[1:])    
    return p

def _pair_probs(x, y):
    return [((x==0)&(y==0)).mean(),
            ((x==0)&(y==1)).mean(),
            ((x==1)&(y==0)).mean(),
            ((x==1)&(y==1)).mean()]

def sisj(eventsx, eventsy, T):
    """Between events listed in x and y DataFrames for x_t and y_{t-1} (i.e. x in the
    future and y in the past). Events are only counted if they occurred within adjacent
    time periods and assumed to take values -1 (no activity) and 1 (activity).
    
    Parameters
    ----------
    eventsx : DataFrame
    eventsy : DataFrame
    T : int
        Length of time series.
    
    Returns
    -------
    ndarray
        Probability for variable pairs (x_{t-1}, y_t), (x_t, y_t), (x_t, y_{t-1}).
        Each row considers all possible pairs of activation in order of (00, 01, 10, 11).
    """
    
    tx = np.zeros(T+1, dtype=int) - 1
    ty = np.zeros(T+1, dtype=int) - 1
    
    if not len(eventsx)==0:
        ix = eventsx.tpixel
        tx[ix] = 1
    if not len(eventsy)==0:
        iy = eventsy.tpixel
        ty[iy] = 1
    
    return (tx[1:] * ty[:-1]).mean()
