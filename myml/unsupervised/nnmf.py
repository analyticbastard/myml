# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:26:50 2013

@author: buhos_000
"""

import numpy as np
import scipy as sp
from scipy import optimize


class SNNMF():
    """
    Implements a sparse non-negative matrix factorization described in [1].
    An imput matrix A is factorized in two factors W and H such that
    ||A-WH||_F^2 + beta*||H||_1 (the sparsity of H) is minimum. The method
    uses the non-negative least squares implemented in Scipy.
    
    [1] Nonnegative Matrix Factorization Based on Alternating Nonnegativity
        Constrained Least Squares and Active Set Method. Hyunsoo Kim and
        Haesun Park. SIAM Journal on Matrix Analysis and Applications, 30-2
    """
    
    def __init__(self, n_components = 2, max_iter = 500,
                 beta = 0.1, eta = 0.1):
        self.n_components_ = n_components
        self.max_iter_ = max_iter
        self.beta_ = beta
        self.eta_ = eta
        
    def fit(self, A):
        n, m = A.shape
        self.nrow_ = n
        self.ncol_ = m
        
        W = np.random.random( (n, self.n_components_) )
        H = self.nnls_H( W, A )
        
        for iter in range( self.max_iter_ ):
            W = self.nnls_W( H, A )
            H = self.nnls_H( W, A )
        
        self.H_ = H
        self.W_ = W
    
    
    def transform(self, A):
        pass
    
    
    def nnls_H(self, W, A):
        e = np.ones( (1, self.n_components_) )
        o = np.zeros( (1, self.ncol_) )
        
        H = np.zeros( (self.n_components_, self.ncol_) )
        
        WW = np.vstack( (W, np.sqrt(self.beta_) * e) )
        AA = np.vstack( (A, o) )
        
        for i in range( self.ncol_ ):
            H[:,i], residual = optimize.nnls( WW, AA[:,i] )
    
        return H
        
    
    def nnls_W(self, H, A):
        I = np.eye( self.n_components_ )
        o = np.zeros( (self.nrow_, self.n_components_) )
        
        W = np.zeros( (self.nrow_, self.n_components_) )
        
        HH = np.vstack( (H.T, np.sqrt(self.eta_) * I) )
        AA = np.vstack( (A.T, o.T) )
        
        for i in range( self.nrow_ ):
            W[i,:], residual = optimize.nnls( HH, AA[:,i] )
    
        return W
        




