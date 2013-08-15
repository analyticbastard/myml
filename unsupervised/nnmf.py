# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:26:50 2013

@author: buhos_000
"""

import numpy as np
import scipy as sp


class SNNMF():
    
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
            H[:,i], residual = sp.optimize.nnls( WW, AA[:,i] )
    
        return H
        
    
    def nnls_W(self, H, A):
        I = np.eye( self.n_components_ )
        o = np.zeros( (self.nrow_, self.n_components_) )
        
        W = np.zeros( (self.nrow_, self.n_components_) )
        
        HH = np.vstack( (H.T, np.sqrt(self.eta_) * I) )
        AA = np.vstack( (A.T, o.T) )
        
        for i in range( self.nrow_ ):
            W[i,:], residual = sp.optimize.nnls( HH, AA[:,i] )
    
        return W
        




