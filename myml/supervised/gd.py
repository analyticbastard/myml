""" Gradient descent module of MyML

    Implementation of several gradient descent variations
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn import preprocessing

import base


class AbstractGradientDescent():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def update_w(self, w, y, ite):
        pass
    
    @abstractmethod
    def get_w(self):
        pass



class Objective():
    __metaclass__ = ABCMeta
    
    @staticmethod
    @abstractmethod
    def f(X, w, y):
        pass
    


class DifferentiableObjective():
    __metaclass__ = ABCMeta
    
    @staticmethod
    @abstractmethod
    def df(X, w, y):
        pass


def gradientDescentFactory(name, fn, **kwargs):
    
    if name == "classical":
        return GradientDescent(fn, **kwargs)
    else: return GradientDescent(fn, **kwargs)
    



class GradientDescent(AbstractGradientDescent):
    """
    Implements the gradient descent approximation algorithm with
    adaptive learning rate.
    
    The objective function and the gradient must be passed to the
    constructor, and must be functions that accept two parameters
    (the current coefficients and the objective vector).
    """
    
    def __init__(self, fn, max_iter = 50, init_eta = 1,
                 init_lr = 0.1, adaptive_lr = True,
                 tol = 10**(-6), scale = True, verbose = True):
        self.fn_obj_   = fn.f
        self.fn_grad_  = fn.df
        self.max_iter_ = max_iter
        self.tol_      = tol
        self.scale_    = scale
        self.verbose_  = verbose
        self.init_eta_ = init_eta
       
       
    def fit(self, X, y):
        n, m = X.shape
        
        if self.scale_:
            self.X_ = preprocessing.scale(X)
            y = preprocessing.scale(y)
        else:
            self.X_ = X
            
        w = np.zeros( (m,1) )
        
        J = self.fn_obj_(X, w, y)
        
        ite = 1
        while ite < self.max_iter_ and np.abs(J) > self.tol_:
            w = self.update_w(w, y, ite)
            J = self.fn_obj_(X, w, y)
            
            if self.verbose_ and ite % 10 == 0:
                print ite, w.T
            
            ite = ite + 1
            
        self.coef_ = w
        self.iter_ = ite
        self.fval_ = J
        
            
    def update_w(self, w, y, ite):
        eta = self.init_eta_/ite
        grad = self.fn_grad_(self.X_, w, y)
        w = w - eta * grad
        return w
        
    
    def get_w(self):
        return self.coef_
        
        

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
    
    