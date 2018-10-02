# 2014-01-04
# Functions for inter state COW data set.
from numpy import *

def get_epnum(startdates,enddates):
    """
        Iterate through all conflicts and when the following conflict begins before the
        previous one ends, put it into the same episode as the previous one and keep
        track of the latest end time in ependdates.
    2014-01-06
    """
    epnum = zeros((len(startdates)))
    ependdate = enddates[0]
    j = 0
    for i in range(1,len(startdates)):
        if (startdates[i]-ependdate).total_seconds()<=0:
            epnum[i] = j
            if (enddates[i]-ependdate).total_seconds()>0:
                ependdate = enddates[i]
        else:
            j += 1
            epnum[i] = j
            ependdate = enddates[i]
#         print i, j, enddates[i],ependdate, (enddates[i]-ependdate).total_seconds()
    return epnum

def remove_neg_dt(startdates,enddates):
    """
    2014-01-05
        Given two list of start times and end times of many episodes, remove cases where
        the time difference from a start time and the end time of the previous event is
        negative, or these events overlap. Replace all events that overlap with one long
        event that has the first start time and the latest end time.
        Value:
            peacelengths : intervals of time in seconds between new set of events
            startdates : start times of new events
            enddates : end ties of new events
            rmix : index of removed events from original inputted vectors
            group : whether removed events belonged to the same group or not
    """
    run = True
    rmix = []
    group = []

    # Get intervals of time in seconds between successive events.
    def get_interv(startdates,enddates):
        interv = []
        for i in range(1,len(enddates)):
            interv.append((startdates[i]-enddates[i-1]).total_seconds())
        interv = array(interv)
        return interv

    peacelengths = get_interv(startdates,enddates)
    i = 0
    j = 0
    while any(peacelengths<0):
        _ix = where(peacelengths<0)[0][0]+1
        if (enddates[_ix-1]-enddates[_ix]).total_seconds()<0:
            enddates[_ix-1] = enddates[_ix]

        # remove bad dates
        __ix = ones((enddates.size))
        __ix[_ix] = 0
        enddates = enddates[__ix==1]
        startdates = startdates[__ix==1]

        peacelengths = get_interv(startdates,enddates)

        # Find groups of conflicts that belong to the same continuous episode. Save the
        # index of those episodes in rmix and the different group labels in group.
        rmix.append(_ix+i)
        if len(rmix)==1:
            group.append(j)
        elif (rmix[-1]-rmix[-2])==1:
            group.append(j)
        else:
            j += 1
            group.append(j)
        i += 1

    return peacelengths,startdates,enddates,array(rmix),array(group)

def print_stats(x,norm=3600*24*365.):
    """
    2014-01-05
        Print basis statistical observations on given data.
        norm : rescale data scalar factor
    """
    import scipy.stats as s

    x = x.copy()
    x /= norm
    print "Normalized by "+str(norm)

    print "n = "+str(x.size)
    print "mean, std, skew = "+str((mean(x),std(x),s.skew(x)))
    print "min, max = "+str((min(x),max(x)))
