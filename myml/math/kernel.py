# -*- coding: utf-8 -*-

# Author: Javier "Analytic Bastard" Arriero-Pais <javierarrieropais@outlook.com>
# Licence: BSD 3 clause

if __name__ == "__main__" and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    __package__ = "myml.math.kernel"


from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn import metrics



    
def kernelFactory(name = "RBF", **kwargs):
    """
    Factory method to anstract the kernel creation 
    """
    
    def returnRBF(**kwargs):
        """
        Auxiliary function to return a Gaussian kernel, since it is
        user more than once 
        """
        try:
            gamma = kwargs["gamma"]
        except:
            gamma = 0.1
        
        return RBFKernel(gamma)
    
    
    if name.lower() == "RBF" or name.lower() == "gaussian":
        return returnRBF(**kwargs)
    elif name.lower() == "linear":
        return LinearKernel()
    else:
        return returnRBF(**kwargs)
    
    


class Kernel:
    """
    Abstract class to define kernel functions and their gradients,
    in case they exist.
    
    Defines a generic interface to retrieve the Gram matrix and the
    gradient matrix and to set the input data to compute both matrices.
    
    Kernel initializations go in their respective constructors.
    """
    __metaclass__ = ABCMeta    
    
    @abstractmethod
    def setData(self, X):
        """
        Sets the input data to compute the Gram matrix and/or the gradient.
        The gram matrix is computed in this step.
        """
        pass
    
    
    def gram(self):
        """
        Returns the gram matrix of the kernel evaluated on the input data
        """
        return self.gram_
    
    
    @abstractmethod
    def gradient(self, i):
        """
        Returns the gradient function of the kernel evaluated on the input data 
        
        We currently support only the Gaussian kernel.
        """
        pass
    
    


class RBFKernel(Kernel):
    """
    Implements Gaussian kernel computation of the Gram matrix and
    the gradient.
    """
    
    def __init__(self, gamma):
        """
        :param gamma: Gamma parameter exp(-gamma ||x_i - x_j||^2)
        """
        self.gamma_ = gamma
       
        
    def setData(self, X):
        self.X_ = X
        self.gram_ = metrics.pairwise_kernels(self.X_, metric = 'rbf',
                                              gamma = self.gamma_)
        
    
    def gradient(self, i):
        """
        Computes the gradient of the Gaussian function at the datum i.
        2*self.gamma_ * (self.X_[i,:] - self.X_[j,:]) * self.gram_[i,j]
        
        :param i: Datum for which to compute the current gradient
        """
        
        n, m = self.X_.shape
        
        g = np.zeros((n,m))
        gamma2 = 2*self.gamma_
        for j in range(n):
            g[j,:] = gamma2 * (self.X_[i,:] - self.X_[j,:]) * self.gram_[i,j]
            
        return g
    
    
class LinearKernel(Kernel):
    def __init__(self):
        pass
    
    def setData(self, X):
        self.X_ = X
        self.gram_ = X.dot(X.T)
        
    def gradient(self, i):
        pass