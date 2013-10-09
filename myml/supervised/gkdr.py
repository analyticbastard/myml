# -*- coding: utf-8 -*-

# Author: Javier "Analytic Bastard" Arriero-Pais <javierarrieropais@outlook.com>
# Licence: BSD 3 clause


if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    __package__ = "myml.supervised.gkdr"

import numpy as np

from sklearn import preprocessing

from ..math import kernel
from .base import AbstractSupervisedMethod

class GKDR(AbstractSupervisedMethod):
    """ Gradient-based Kernel Dimension Reduction
        
        The method can be used as supervised learning by using a guiding
        variable y, or as unsupervised learning by using the same variable
        X as an autoregressor.
    
        The only supported kernel is the Gaussian (RBF) kernel for its
        characteristic property and its differentiability at all points.
        
        References:
        [1] K. Fukumizu, C. Leng - Gradient-based kernel method for feature 
            extraction and variable selection. NIPS 2012.
    """

    def __init__(self, n_components = 2, scale = True, 
                 gamma_x = 1, gamma_y = 1, lamb = 0.01):
        """
            :param n_components: The number of dimensions in the reduced 
            representation. Detault 2.
            :param scale: Whether to scale data. Default True.
            :param gamma_x: Gamma parameter for the kernel on X
            :param gamma_y: Gamma parameter for the kernel on y
            :param lamb: Regularization constant for the kernel on X inversion
        """
        self.n_components_ = n_components
        
        self.scale_   = scale
        self.ky_      = kernel.kernelFactory("RBF", gamma_x)
        self.kx_      = kernel.kernelFactory("RBF", gamma_y)
        self.lamb_    = lamb
        
    
        
        
    def fit(self, X, y = None):
        """ Computers the reduced dimension from X and (possibly) y.
        
            :param X: Data covariates variable X to reduce the dimensionality
            :param y: Guiding variable. Optional (using X as autoregressor).
        """
        if self.scale_:
            X = preprocessing.scale(X)
            
            if y:
                if y.shape != X.shape:
                    return
                    
                y = preprocessing.scale(y)
                
            else:
                y = X
        
        n, m = X.shape
        lam     = self.lamb_
        
        self.ky_.setX(y)
        self.kx_.setX(X)
        
        Ky = self.ky_.gram()
        Kx = self.kx_.gram()
        
        Gx = Kx + lam*Kx.shape[0]*np.eye(Kx.shape[0])
        Gy = Ky
        
        IGx = np.linalg.inv(Gx)
        G   = np.dot(IGx, Gy)
        G   = np.dot(G, IGx)
        
        M = np.zeros((m,m))
        for i in range(n):
            gx = self.kx_.gradient(i)
            M = M + np.dot(gx.T, np.dot(G, gx) )
            
        eb, U = np.linalg.eigh(M)
        
        self.weights_ = eb
        self.coef_    = U[:,-self.n_components_: ]
        
    
    
    def predict(self, X):
        return X.dot( self._coef_ )
        