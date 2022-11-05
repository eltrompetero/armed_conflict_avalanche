# ====================================================================================== #
# Module for computing self transfer entropy for polygons from conflict data.
# 
# Author : Niraj Kushwaha, kushwaha@csh.ac.at
# Edited : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .utils import *

def iter_valid_polygons(valid_polygons, number_of_shuffles, time_series , n_cpu=None):
    """Calculates self transfer entropy for each valid polygon's time series and shuffles.

    Parameter
    ---------
    valid_polygons : list
    number_of_shuffles : pd.Dataframe
    n_cpu : int, None

    Returns
    -------
    dict
        Key is polygon index. Value is list (TE, TE shuffles).
    
    """

    SelfTEcalculator = SelfTransferEntropy()


    def loop_wrapper(pol):
        x = time_series[[pol]].values
        return pol, (SelfTEcalculator.calc(x),
                        [SelfTEcalculator.shuffle_calc(x) for i in range(number_of_shuffles)])

    if n_cpu is None:
        pol_te = {}
        for pol in valid_polygons:
            pol_te[pol] = loop_wrapper(pol)[1]
    else:
        with Pool() as pool:
            pol_te = dict(pool.map(loop_wrapper, valid_polygons , chunksize=100))
    return pol_te



class SelfTransferEntropy():
    def __init__(self , rng=None):
        """
        Parameters
        ----------
        rng : np.random.RandomState, None
        """

        self.important_indexes = [(0,0,0),(1,1,0),(2,0,1),(3,1,1)]

        self.rng = np.random if rng is None else rng

    def calc(self, x):
        """Calculate self loop transfer entropy for time series of a tile.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        float
            TE in bits.
        """

        assert set(np.unique(x)) <= frozenset((0,1)), "x must be binary"

        time_series = x  #Recheck this

        joint_distribution_xT_xt_list = joint_distribution_xT_xt_new(time_series)
        distribution_xt_list = distribution_xt_new(time_series)
        distribution_xT_list = distribution_xT_new(time_series)

        te_terms = []

        for i,j,k in self.important_indexes:
            if(joint_distribution_xT_xt_list[i] == 0 or 
                distribution_xt_list[j] == 0 or 
                distribution_xT_list[k] == 0):
                    pass
            else:
                te_terms.append(self_loop_te_calculator((joint_distribution_xT_xt_list[i],
                                                            distribution_xt_list[j],
                                                            distribution_xT_list[k])))
        
        return sum(te_terms)

    def shuffle_calc(self, x):
        """Shuffle time series and return naive transfer entropy estimate.

        Returns
        -------
        float
            TE in bits.
        """

        randix = self.rng.permutation(np.arange(x.size, dtype=int))
        return self.calc(x[randix])
# end SelfTransferEntropy





@njit()
def joint_distribution_xT_xt_new(time_series_two_pols):
    counts_array = np.zeros(4)   #(xT,xt)  0->00 , 1->01 , 2->10 , 3-> 11

    time_series_xT_xt = time_series_two_pols
    #time_series_xT_xt = time_series_xT_xt[:-1]

    time_series_length = len(time_series_xT_xt)


    for i in range(time_series_length-1):           
        if(time_series_xT_xt[i+1][0] == 0 and time_series_xT_xt[i][0] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_xT_xt[i+1][0] == 0 and time_series_xT_xt[i][0] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_xT_xt[i+1][0] == 1 and time_series_xT_xt[i][0] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_xT_xt[i+1][0] == 1 and time_series_xT_xt[i][0] == 1):
            counts_array[3] = counts_array[3] + 1

    distribution_array = np.zeros(4)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / (time_series_length-1)

    return distribution_array


@njit()
def distribution_xt_new(time_series_two_pols):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_x = time_series_two_pols[:,0]

    #time_series_x = time_series_x[1:]
    time_series_x = time_series_x[:-1]

    time_series_length = len(time_series_x)

    for i in range(len(time_series_x)):
        if(time_series_x[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_x[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array


@njit()
def distribution_xT_new(time_series_two_pols):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_x = time_series_two_pols[:,0]

    #time_series_x = time_series_x[1:]
    time_series_x = time_series_x[1:]

    time_series_length = len(time_series_x)

    for i in range(len(time_series_x)):
        if(time_series_x[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_x[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array

def self_loop_te_calculator(distribution_data):
    xT_xt , xt , xT = distribution_data
    return (xT_xt * np.log2((xT_xt) / (xt*xT)))