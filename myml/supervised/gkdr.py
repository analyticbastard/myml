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
    """
    Gradient-based Kernel Dimension Reduction
        
    The method can be used as supervised learning by using a guiding
    variable y, or as unsupervised learning by using the same variable
    X as an autoregressor.

    The only supported kernel is the Gaussian (RBF) kernel for its
    characteristic property and its differentiability at all points.
    This is done with the help of the abstracted kernel abstractions
    under the math package, which return the gradients.
    
    References:
    [1] K. Fukumizu, C. Leng - Gradient-based kernel method for feature 
        extraction and variable selection. NIPS 2012.
    """

    def __init__(self, n_components = 2, scale = True, 
                 lamb = 0.001, **kwargs):
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
        self.lamb_    = lamb
        
        # The following dicts are the parameters for kx and ky kernels
        yargs = dict()
        xargs = dict()
        
        # Gaussian by default
        yname = "RBF"
        xname = "RBF"
        
        # We extract the possible names for both kernels, and its parameters
        for a in kwargs:
            if a == "kernel_y":
                yname = kwargs[a]
            elif a == "kernel_x":
                xname = kwargs[a]    
            elif a.endswith("_y"):
                # We remove the trailing _y for it to be subsequently valid
                yargs[a[:-2]] = kwargs[a]
            elif a.endswith("_x"):
                # We remove the trailing _x for it to be subsequently valid
                xargs[a[:-2]] = kwargs[a]
        
        self.ky_ = kernel.kernelFactory(yname, **yargs)
        self.kx_ = kernel.kernelFactory(xname, **xargs)
        
    
        
        
    def fit(self, X, y = None):
        """
        Computers the reduced dimension from X and (possibly) y.
        
        :param X: Data covariates variable X to reduce the dimensionality
        :param y: Guiding variable. Optional (using X as autoregressor).
        """
        if self.scale_:
            X = preprocessing.scale(X)
            
            if y != None:
                if y.shape[0] != X.shape[0]:
                    # Throw exception
                    return
                    
                y = preprocessing.scale(y)
                
            else:
                y = X
        
        n, m = X.shape
        lam  = self.lamb_
        
        self.ky_.setData(y)
        self.kx_.setData(X)
        
        Ky = self.ky_.gram()
        Kx = self.kx_.gram()
        
        Gx = Kx + lam*Kx.shape[0]*np.eye(Kx.shape[0])
        Gy = Ky
        
        IGx = np.linalg.inv(Gx)
        G   = np.dot(IGx, Gy)
        G   = np.dot(G, IGx)
        
        M = None
        for i in range(n):
            gx = self.kx_.gradient(i)
            
            Mi = np.dot(gx.T, np.dot(G, gx) )
            if M == None: M = Mi
            else: M = M + Mi
        
        M = M/n
        eb, U = np.linalg.eigh(M)
        
        self.M_       = M
        self.weights_ = eb
        self.coef_    = U[:,-self.n_components_: ]
        
    
    
    def predict(self, X):
        return X.dot( self.coef_ )
        