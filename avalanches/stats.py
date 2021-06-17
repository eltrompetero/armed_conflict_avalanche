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
def joint_prob(eventsx, eventsy, T,
               cond_t_bds=None):
    """Joint probabilities between events listed in x and y DataFrames. Only considering
    whether any event occurred at all or not. Events are only counted if they occurred
    within adjacent time periods.
    
    Parameters
    ----------
    eventsx : DataFrame
        Must have a tpixel coordinate that can start from 1 and go up to including T.
    eventsy : DataFrame
    T : int
        Length of time series. This is necessary b/c the full length of the time series
        cannot be known from the event list, which may not go to the max possible time.
    cond_t_bds : list of twoples, None
        Option to condition calculation to certain ranges of tpixel. Bounds are inclusive.
    
    Returns
    -------
    ndarray
        Probability for variable pairs (x_{t-1}, y_t), (x_t, y_t), (x_{t+1}, y_t).
        Each row considers all possible pairs of activation in order of (00, 01, 10, 11).
    int (optional)
        If cond_t_bds is specified, then the total duration of samples used to calculate
        the result is returned to be able to better assess accuracy.
    """

    assert isinstance(eventsx, pd.DataFrame) and isinstance(eventsy, pd.DataFrame)
    assert T>1
    
    if not cond_t_bds is None:
        # recursively call joint_prob on visible segments and weighted aggregate at end
        plist, Tlist = [], []

        # find each visible starting segment
        for t0, t1 in cond_t_bds:
            if (t1-t0)>1:
                # select all events that fall within the time window
                eventsx_ = eventsx.iloc[(eventsx.tpixel>=t0).values&(eventsx.tpixel<=t1).values]
                eventsy_ = eventsy.iloc[(eventsy.tpixel>=t0).values&(eventsy.tpixel<=t1).values]
                # restart pixel count to begin and end with the specified time window
                eventsx_.tpixel -= t0 - 1
                eventsy_.tpixel -= t0 - 1

                Tlist.append(t1-t0+1)
                plist.append(joint_prob(eventsx_, eventsy_, Tlist[-1]))
        
        totT = sum(Tlist) - len(Tlist)
        # weighted average, remember that each term only involves T-1 entries
        avgp = sum([p_*(T_-1)/totT for p_, T_ in zip(plist, Tlist)])
        return avgp, totT
    
    # when no cond_t_bds is given...
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

def sisj(eventsx, eventsy, T, dt=1, laplace_prior=False):
    """Between events listed in x and y DataFrames for x_t and y_{t-dt} (i.e. x in the
    future and y in the past). Events are only counted if they occurred within adjacent
    time periods and assumed to take values -1 (no activity) and 1 (activity).
    
    Parameters
    ----------
    eventsx : DataFrame
        Events in future. 
    eventsy : DataFrame
        Events in past.
    T : int
        Time point of latest event, i.e. value of time at last event given counting starts
        with 0.
    dt : int, 1
    laplace_prior : bool, False
        If True, account for four additional possible configurations.
    
    Returns
    -------
    float
    """
    
    assert isinstance(dt, int) and dt>=0
    tx = np.zeros(T+1, dtype=int) - 1
    ty = np.zeros(T+1, dtype=int) - 1
    
    # if at least one event is in eventsx
    if not len(eventsx)==0:
        ix = eventsx.tpixel
        tx[ix] = 1
    # if at least one event is in eventsy
    if not len(eventsy)==0:
        iy = eventsy.tpixel
        ty[iy] = 1
    
    if laplace_prior:
        tx = np.concatenate((tx, [-1,-1,1,1]))
        ty = np.concatenate((ty, [-1,1,-1,1]))

    # return pair correlation
    if dt==0:
        return (tx * ty).mean()
    return (tx[dt:] * ty[:-dt]).mean()

def sijk(eventsx, eventsy, eventsz, T, dt=1, laplace_prior=False):
    """Between events listed in x, y, and z DataFrames for x_t, y_{t-dt}, and z_{t-dt}
    (i.e. x in the future and y and z in the past). Events are only counted if they
    occurred within adjacent time periods and assumed to take values -1 (no activity) and
    1 (activity).
    
    Parameters
    ----------
    eventsx : DataFrame
    eventsy : DataFrame
    eventsz : DataFrame
    T : int
        Length of time series, i.e. value of time at last event where counting starts with
        0.
    dt : int, 1
    laplace_prior : bool, False
    
    Returns
    -------
    float
    """
    
    assert isinstance(dt, int) and dt>=0
    tx = np.zeros(T+1, dtype=int) - 1
    ty = np.zeros(T+1, dtype=int) - 1
    tz = np.zeros(T+1, dtype=int) - 1
    
    # if at least one event is in eventsx
    if not len(eventsx)==0:
        ix = eventsx.tpixel
        tx[ix] = 1
    # if at least one event is in eventsy
    if not len(eventsy)==0:
        iy = eventsy.tpixel
        ty[iy] = 1
    # if at least one event is in eventsz
    if not len(eventsz)==0:
        iz = eventsz.tpixel
        tz[iz] = 1
    
    if laplace_prior:
        tx = np.concatenate((tx, [-1,-1,-1,-1,1,1,1,1]))
        ty = np.concatenate((ty, [-1,-1,1,1,-1,-1,1,1]))
        tz = np.concatenate((tz, [-1,1,-1,1,-1,1,-1,1]))

    # return pair correlation
    if dt==0:
        return (tx * ty * tz).mean()
    return (tx[dt:] * ty[:-dt] * tz[:-dt]).mean()

