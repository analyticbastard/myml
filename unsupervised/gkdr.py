"""Kernel Dimension Reduction.

.. moduleauthor:: Javier Arriero-Pais <javierarrieropais@outlook.com>

"""

import numpy as np
from sklearn import preprocessing, metrics

class GKDR():
    """ Gradient-based Kernel Dimension Reduction
    
        The only supported kernel is the Gaussian (RBF) kernel for its
        characteristic property and its differentiability at all points.
        
        The method can be used as supervised learning by using a guiding
        variable y, or as unsupervised learning by using the same variable
        X as an autoregressor.
        
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
        self.gamma_x_ = gamma_x
        self.gamma_y_ = gamma_y
        self.lamb_     = lamb
        
        
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
        
        gamma_y = self.gamma_x_
        gamma_x = self.gamma_y_
        lam     = self.lamb_
        
        Ky = metrics.pairwise_kernels(y, metric='rbf', gamma= gamma_y)
        Kx = metrics.pairwise_kernels(X, metric='rbf', gamma= gamma_x)
        
        Gx = Kx + lam*Kx.shape[0]*np.eye(Kx.shape[0])
        Gy = Ky
        
        IGx = np.linalg.inv(Gx)
        G   = np.dot(IGx , Gy)
        G   = np.dot(G, IGx)
        
        M = np.zeros((m,m))
        for i in range(n):
            gx = self.rbf_gradient(Kx, gamma_x, X, i)
            
            M = M + np.dot(gx.T, np.dot(G, gx) )
            
        eb, B = np.linalg.eigh(M)
        
        Beta = B[:,eb.argmax()[-self.n_components_:] ]
        
        return Beta
        
        
    
    def rbf_gradient(K, gamma, X, n, m, i):
        """ Computes the gradient of the Gaussian function
        
            :param K: RBF kernel on the variable :param X:
            :param X: Data X corresponding to the kernel matrix :param K:
            :param i: Datum for which to compute the current gradient
        """
        n, m = X.shape
        
        g = np.zeros((n,m))
        for j in range(n):
            g[j,:] = 2*gamma * (X[i,:] - X[j,:]) * K[i,j]
            
        return g
        