# ====================================================================================== #
# Module for computing transfer entropy between polygons from conflict data.
# 
# Author : Niraj Kushwaha
# Edited : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from .utils import *


def iter_polygon_pair(polygon_pair, number_of_shuffles, time_series,
                      type_of_TE='n',
                      n_cpu=None):
    """Calculate transfer entropy for pairs of polygon time series and shuffles.

    Parameters
    ----------
    number_of_shuffles : int
    time_series : pd.DataFrame
    type_of_TE : str, 'n'
    n_cpu : int, None
    
    Returns
    -------
    dict
        Key is directed edge btwn polygons. Value is list (TE, TE shuffles).
    """

    TEcalculator = TransferEntropy(type_of_TE)

    # calculate transfer entropy for a single pair of tiles
    def loop_wrapper(pair):
        x = time_series[pair[0]].values
        y = time_series[pair[1]].values
        return pair, (TEcalculator.calc(x, y),
                      [TEcalculator.shuffle_calc(x, y) for i in range(number_of_shuffles)])
    
    if n_cpu is None:
        pair_te = {}
        for pair in polygon_pair:
            pair_te[pair] = loop_wrapper(pair)[1]
    else:
        # note that each job is quite fast, so large chunks help w/ speed
        with Pool() as pool:
            pair_te = dict(pool.map(loop_wrapper, polygon_pair, chunksize=200))
    return pair_te



# ================================= #
# Classes and their special methods #
# ================================= #
class TransferEntropy():
    def __init__(self, type_of_TE, rng=None):
        """
        Parameters
        ----------
        type_of_TE : str
        rng : np.random.RandomState, None
        """

        if(type_of_TE == "n"):
            self.important_indexes = [(0,0,0,0),(1,0,1,0),(2,1,2,1),(3,1,3,1),(4,2,0,0),(5,2,1,0),(6,3,2,1),(7,3,3,1)]
            self.TE_type = "transfer_entropy"
            self.definition_type = ""

        elif(type_of_TE == "ep"):
            self.important_indexes = [(0,0,0,0),(2,1,2,1),(5,2,1,0),(7,3,3,1)]
            self.TE_type = "exci"
            self.definition_type = "paper_"

        elif(type_of_TE == "eo"):
            self.important_indexes = [(1,0,1,0),(3,1,3,1),(5,2,1,0),(7,3,3,1)]
            self.TE_type = "exci"
            self.definition_type = "our_"

        elif(type_of_TE == "ip"):
            self.important_indexes = [(1,0,1,0),(3,1,3,1),(4,2,0,0),(6,3,2,1)]
            self.TE_type = "inhi"
            self.definition_type = "paper_"

        elif(type_of_TE == "io"):
            self.important_indexes = [(0,0,0,0),(2,1,2,1),(4,2,0,0),(6,3,2,1)]
            self.TE_type = "inhi"
            self.definition_type = "our_"
        else:
            raise NotImplementedError("Bad value for type_of_TE.")

        self.rng = np.random if rng is None else rng

    def calc(self, x, y):
        """Calculate transfer entropy between two binary time series.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        float
            TE in bits.
        """
        
        assert set(np.unique(x)) <= frozenset((0,1)), "x must be binary"
        assert set(np.unique(y)) <= frozenset((0,1)), "y must be binary"

        time_series = np.vstack((x, y)).T
        
        # calculate each term in the transfer entropy equation and sum them together
        joint_distribution_yT_yt_xt_list = joint_distribution_yT_yt_xt_new(time_series)
        joint_distribution_yT_yt_list = joint_distribution_yT_yt_new(time_series)
        joint_distribution_yt_xt_list = joint_distribution_yt_xt_new(time_series)
        distribution_y_list = distribution_y_new(time_series)

        te_terms = []

        for i, j, k, l in self.important_indexes:
            if(joint_distribution_yT_yt_xt_list[i] == 0 or
               joint_distribution_yT_yt_list[j] == 0 or
               joint_distribution_yt_xt_list[k] == 0 or
               distribution_y_list[l] == 0):
                pass
            else:
                te_terms.append(te_calculator((joint_distribution_yT_yt_xt_list[i],
                                               joint_distribution_yT_yt_list[j],
                                               joint_distribution_yt_xt_list[k],distribution_y_list[l])))
        return sum(te_terms)
    
    def shuffle_calc(self, x, y):
        """Shuffle both time series in the same way and return naive transfer entropy estimate.

        Returns
        -------
        float
            TE in bits.
        """

        randix = self.rng.permutation(np.arange(x.size, dtype=int))
        return self.calc(x[randix], y[randix])
# end TransferEntropy


@njit
def distribution_y_new(time_series_two_pols):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_y = time_series_two_pols[:,1]

    #time_series_y = time_series_y[1:]
    time_series_y = time_series_y[:-1]

    time_series_length = len(time_series_y)

    for i in range(len(time_series_y)):
        if(time_series_y[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_y[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array

@njit()
def distribution_x_new(time_series_two_pols):
    counts_array = np.zeros(2)   #0->0 , 1->1

    time_series_y = time_series_two_pols[:,0]

    #time_series_y = time_series_y[1:]
    time_series_y = time_series_y[:-1]

    time_series_length = len(time_series_y)

    for i in range(len(time_series_y)):
        if(time_series_y[i] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_y[i] == 1):
            counts_array[1] = counts_array[1] + 1

    distribution_array = np.zeros(2)

    distribution_array[0] = counts_array[0] / time_series_length
    distribution_array[1] = counts_array[1] / time_series_length

    return distribution_array

@njit()
def joint_distribution_yt_xt_new(time_series_two_pols):
    counts_array = np.zeros(4)   #(yt,xt)  0->00 , 1->01 , 2->10 , 3-> 11

    #time_series_yt_xt = time_series_two_pols[1:]
    time_series_yt_xt = time_series_two_pols[:-1]

    time_series_length = len(time_series_yt_xt)

    for i in range(time_series_length):           
        if(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yt_xt[i][1] == 0 and time_series_yt_xt[i][0] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yt_xt[i][1] == 1 and time_series_yt_xt[i][0] == 1):
            counts_array[3] = counts_array[3] + 1

    distribution_array = np.zeros(4)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / time_series_length

    return distribution_array

@njit()
def joint_distribution_yT_yt_new(time_series_two_pols):
    counts_array = np.zeros(4)   #(yT,yt)  0->00 , 1->01 , 2->10 , 3-> 11

    time_series_yT_yt = time_series_two_pols
    #time_series_yT_yt = time_series_yT_yt[:-1]

    time_series_length = len(time_series_yT_yt)


    for i in range(time_series_length-1):           
        if(time_series_yT_yt[i+1][1] == 0 and time_series_yT_yt[i][1] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yT_yt[i+1][1] == 0 and time_series_yT_yt[i][1] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yT_yt[i+1][1] == 1 and time_series_yT_yt[i][1] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yT_yt[i+1][1] == 1 and time_series_yT_yt[i][1] == 1):
            counts_array[3] = counts_array[3] + 1

    distribution_array = np.zeros(4)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / (time_series_length-1)

    return distribution_array

@njit
def joint_distribution_yT_yt_xt_new(time_series_two_pols):
    counts_array = np.zeros(8)   #(yT,yt,xt)  0->000,1->001,2->010,3->011,4->100,5->101,6->110,7->111

    time_series_yT_yt_xt = time_series_two_pols
    #time_series_yT_yt_xt = time_series_yT_yt_xt[:-1]

    time_series_length = len(time_series_yT_yt_xt)

    for i in range(time_series_length-1):           
        if(time_series_yT_yt_xt[i+1][1] == 0 and
           time_series_yT_yt_xt[i][1] == 0 and
           time_series_yT_yt_xt[i][0] == 0):
            counts_array[0] = counts_array[0] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 0 and
             time_series_yT_yt_xt[i][1] == 0 and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[1] = counts_array[1] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 0 and
             time_series_yT_yt_xt[i][1] == 1 and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[2] = counts_array[2] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 0  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[3] = counts_array[3] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 1  and
             time_series_yT_yt_xt[i][1] == 0  and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[4] = counts_array[4] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 1  and
             time_series_yT_yt_xt[i][1] == 0  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[5] = counts_array[5] + 1
        elif(time_series_yT_yt_xt[i+1][1] == 1  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 0):
            counts_array[6] = counts_array[6] + 1              
        elif(time_series_yT_yt_xt[i+1][1] == 1  and
             time_series_yT_yt_xt[i][1] == 1  and
             time_series_yT_yt_xt[i][0] == 1):
            counts_array[7] = counts_array[7] + 1            

    distribution_array = np.zeros(8)

    for i in range(len(distribution_array)):
        distribution_array[i] = counts_array[i] / (time_series_length-1)

    return distribution_array

def te_calculator(distribution_data):
    yT_yt_xt , yT_yt , yt_xt , y = distribution_data
    return (yT_yt_xt * np.log2((yT_yt_xt * y) / (yT_yt * yt_xt)))

