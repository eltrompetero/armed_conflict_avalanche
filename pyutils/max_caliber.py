# ====================================================================================== #
# Module for running max caliber inference on local site dynamics.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from coniii.solvers import SparseEnumerate, SparseMCH
from coniii.utils import define_ising_helper_functions, bin_states
from .utils import *



class Inference11():
    """Inferring Markov chain model of single-spin to single-spin correlations with time
    depth of 1.
    """
    def __init__(self, X):
        """
        Parameters
        ----------
        X : ndarray
        """

        self.X = X
        self.n = X.shape[1]
        self.calc_e = define_ising_helper_functions()[0]
        self.corr = self.one_to_one_corr(X) 
    
    def solve(self, threshold_n=8, sample_kw={'sample_size':10_000}):
        """Find maxent parameters.

        Will automatically switch from an expensive Enumerate technique to an MCH
        technique above threshold n.

        Parameters
        ----------
        threshold_n : int, 8
        sample_kw : dict, {'sample_size':10_000}


        Returns
        -------
        ndarray
        """
        
        if self.n <= threshold_n:
            # using SparseEnumerate to set up an example system and test calculations
            # effectively, must set up a 2*n spin system with couplings across time
            solver = SparseEnumerate(2 * self.n, parameter_ix=self.one_to_one_parameter_ix(self.n))
            params = solver.solve(constraints=self.corr)
        
        else:
            # using SparseMCH to set up an example system and test calculations
            # effectively, must set up a 2*n spin system with couplings across time
            solver = SparseMCH(2 * self.n,
                               parameter_ix=one_to_one_parameter_ix(self.n),
                               **sample_kw)



            params = solver.solve(constraints=self.corr,
                                  custom_convergence_f=self.learn_settings,
                                  maxiter=30)

        return params

    # Define function for changing learning parameters as we converge.
    def learn_settings(self, i):
        """Take in the iteration counter and set the maximum change allowed in any given
        parameter (maxdlamda) and the multiplicative factor eta, where d(parameter) =
        (error in observable) * eta.
        
        Additional option is to also return the sample size for that step by returning a
        tuple. Larger sample sizes are necessary for higher accuracy.
        """
        return {'maxdlamda':exp(-i/5.),'eta':exp(-i/5.)}

    @staticmethod
    def one_to_one_corr(X):
        """Correlations from single spins to single spins in the future.
        
        Parameters
        ----------
        X : ndarray
        
        Returns
        -------
        ndarray
            Correlations are organized such that we look at <x_t * y_{t+1}>, and we
            iterate thru every y first for every x.
        """
        
        L = X.shape[0]
        N = X.shape[1]
        corr = np.zeros(N*N)
        
        counter = 0
        for i in range(N):
            for j in range(N):
                corr[counter] = X[:-1,i].dot(X[1:,j]) / L
                counter += 1
        return corr
    
    @staticmethod
    def one_to_one_parameter_ix(n):
        """
        Parameters
        ----------
        n : int
        
        Returns
        -------
        ndarray of int
        """
        
        interactionMat = np.zeros((2*n, 2*n), dtype=int)
        interactionMat[:n,n:] = 1
        couplingIx = np.where(interactionMat[np.triu_indices(2*n, k=1)])[0] + 2 * n
        return couplingIx
#end Inference11
