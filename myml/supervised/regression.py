# -*- coding: utf-8 -*-

# Author: Javier "Analytic Bastard" Arriero-Pais <javierarrieropais@outlook.com>
# Licence: BSD 3 clause


if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    __package__ = "myml.supervised.gkdr"

import numpy as np

from scipy   import linalg

from ..math  import kernel
from .base   import AbstractSupervisedMethod
import classification
import gd

from ..unsupervised import utils





class Square(gd.DifferentiableObjective):
    """
    Square loss objective function and gradient
    
    Provides the function and gradient for an
    AbstractGradientDescent method to work
    """
    @staticmethod
    def f(X, w, y):
        """
        Objective function for Squared loss (Linear Regression)
        """
        return np.sum( (y - X.dot(w) )**2 )
    
    
    @staticmethod
    def df(X, w, y):
        """
        Gradient for Squared Loss function
        """
        N, M = X.shape
        vec = np.zeros((N,M))
        for i in range(N):
            vec[i,:] = - (y[i] - X[i,:].dot(w) )*X[i,:]
        
        vec = vec.sum(axis=0)
        return vec.reshape((M,1))
    



class OLS(AbstractSupervisedMethod):
    """
    Least squares or multiple linear regression with a gradient
    descent estimation of the parameters.
    """
    def __init__(self, gd_name = "classical", intercept = True, **kwargs):
        self.gd_ = gd.gradientDescentFactory(gd_name, Square, **kwargs)
        self.intercept_  = True
    
    
    
    def fit(self, X, y):
        X_ = classification.getData(self.intercept_, X)
            
        self.gd_.fit(X_, y)
        pass
    
    
    def predict(self, X):
        X_ = classification.getData(self.intercept_, X)
        w = self.gd_.get_w()
        
        return X_.dot(w)





class NNLS(AbstractSupervisedMethod):
    """
    Non-negative least squares
    
    [1] Lawson C., Hanson R.J., Solving Least Squares Problems, SIAM. 1987
    """
    
    def __init__(self, max_iter = 500):
        self.max_iter_ = max_iter
        
    
    def fit(self, X, y):
        n, m = X.shape
        self.nrow_ = n
        self.ncol_ = m
        
        g = np.zeros( (m, 1) )
        
        self.E = range( n )
        self.S = []
        
        w = X.T.dot ( y - X.dot(g) )
        
        while len(self.E)>0 and np.any(w>0):
            self.move_E_S(X, y, g)
            
            Bs = np.zeros( (n, m) )
            Bs[:,self.S] = X[:, self.S ]
            
            z = linalg.lstsq(Bs, y)[0]
            print z
            z[self.E] = 0
            
            while np.any( z[self.S] <= 0 ):
                alpha = np.min( g/(g-z) )
                g = g + alpha * ( z - g )
                
                self.move_S_E( g )
                
                Bs = np.zeros( (n, m) )
                Bs[:,self.S] = X[:, self.S ]
                
                z = linalg.lstsq(Bs, y)[0]
                z[self.E] = 0
            
            self.coef_ = g = z
            
            w = X.T.dot( y - X.dot(g) )
    
        
        
    def predict(self, X):
        return X.dot( self.coef_ )
        
            
        
    def find_t_max(self, X, y, g):
        max_val = 0
        ind = -1
        
        for i in range(self.ncol_):
            b = X[:,i:(i+1)]
            w = ( y - X.dot(g) ).dot(b)[0]
            
            if max_val < w:
                max_val = w
                ind = i
                
        return [ind]
        
    
    def move_E_S(self, X, y, g):
        ind = self.find_t_max(X, y, g)
        if ind == -1:
            return
            
        print ind, self.E
        self.E = np.setdiff1d(self.E, ind).tolist()
        self.S = np.setdiff1d(self.S, ind).tolist()
        self.S.extend(ind)
        
        
    def move_S_E(self, g):
        ind = np.where( g < 10**(-6) )[0]
        
        idx = np.intersect1d( self.S, ind ).tolist()
        self.S = np.setdiff1d( self.S, idx ).tolist()
        self.E = np.setdiff1d( self.E, idx ).tolist()
        self.E.extend( idx )
        
        


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
        :param y: Dependent variable. Optional (using X as autoregressor).
        """
        if self.scale_:
            self.scalerx_ = utils.Scale()
            self.scalerx_.fit(X)
            X = self.scalerx_.transform(X)
            
            if y != None:
                if y.shape[0] != X.shape[0]:
                    # Throw exception
                    return
                    
                self.scalery_ = utils.Scale()
                self.scalery_.fit(y)
                y = self.scalery_.transform(y)
            
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
        if self.scale_:
            X = self.scalerx_.transform(X)
                
        return X.dot( self.coef_ )
        
        
