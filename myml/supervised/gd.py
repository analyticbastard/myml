""" Gradient descent module of MyML

    Implementation of several gradient descent variations
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn import preprocessing

import base


class AbstractGradientDescent(base.AbstractSupervisedMethod):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def update_w(self, w, y, ite):
        pass



class GradientDescent(AbstractGradientDescent):
    """
    Implements the gradient descent approximation algorithm with
    adaptive learning rate.
    
    The objective function and the gradient must be passed to the
    constructor, and must be functions that accept two parameters
    (the current coefficients and the objective vector).
    """
    
    def __init__(self, fn_obj, fn_grad, max_iter = 500,
                 init_lr = 0.1, adaptive_lr = True,
                 tol = 10**(-6), scale = True):
        self.fn_obj_ = fn_obj
        self.fn_grad_ = fn_grad
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.scale_ = scale
       
       
    def fit(self, X, y):
        n, m = X.shape
        
        if self.scale_:
            X = preprocessing.scale(X)
            y = preprocessing.scale(y)
            
        w = np.zeros( (m,1) )
        
        J = self.fn_obj_(w, y)
        
        ite = 1
        while ite < self.max_iter_ and J > self.tol_:
            w = self.update_w(w, y, ite)
            J = self.fn_obj_(w, y)
            
        self.coef_ = w
        
    
    def predict(self, X):
        pass
            
            
    def update_w(self, w, y, ite):
        eta = 1/ite
        
        w = w - eta * self.fn_grad_(w, y)
        
        

class StochasticGradientDescent(AbstractGradientDescent):
    def __init__(self, fn_obj, fn_grad, max_iter = 500,
                 tol = 10**(-6), scale = True):
        self.fn_obj_ = fn_obj
        self.fn_grad_ = fn_grad
        self.max_iter_ = max_iter
        self.tol_ = tol
        self.scale_ = scale
        
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    