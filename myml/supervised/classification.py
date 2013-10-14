'''
Created on 13/10/2013

@author: Javier Arriero Pais
'''


import numpy as np

import base
import gd




def getData(intercept, X):
    if intercept:
        N = X.shape[0]
        X_ = np.hstack( (np.ones((N,1)), X) )
    else:
        X_ = X
        
    return X_
    

def logistic(x):
    """
    Logistic function f(x) = 1/(1+np.exp(-x))
    """
    return 1/(1+np.exp(-x))



class Logistic(gd.DifferentiableObjective):
    """
    Logistic objective function and gradient
    
    Provides the function and gradient for an
    AbstractGradientDescent method to work
    """
    
    @staticmethod
    def logistic(X, w):
        """
        Logistic function for the scalar product between the
        parameters and the variables logistic(Xw) in vector form
        """
        N = X.shape[0]
        vec = X.dot(w)
        for i in range(N):
            vec[i] = logistic( vec[i] )
            
        return vec
    
    
    @staticmethod
    def f(X, w, y):
        """
        Objective function for Logistic Regression
        """
        lxw = Logistic.logistic( X, w )
        term1 = np.sum( y - np.log(lxw) )
        term2 = np.sum( (1-y) * np.log(1-lxw) )
        return term1 + term2
    
    
    @staticmethod
    def df(X, w, y):
        """
        Gradient for Logistic Regression
        """
        N, M = X.shape
        vec = np.zeros((N,M))
        for i in range(N):
            vec[i,:] = - (y[i] - logistic(X[i,:].dot(w)) )*X[i,:]
        
        vec = vec.sum(axis=0)
        return vec.reshape((M,1))



class LogisticRegression(base.AbstractSupervisedMethod):
    """
    Logistic Regression
    Implements logistic regression for classification by calling a
    gradient descent method with the Logistic function class, which
    provides the objective function and gradient that the GD method
    needs.
    
    [1] Stanford CS 229 Lecture Notes (Logistic Regression), Andrew Ng
    """
    def __init__(self, gd_name = "classical", intercept = True, **kwargs):
        self.gd_ = gd.gradientDescentFactory(gd_name, Logistic, **kwargs)
        self.intercept_  = True
    
    
    def fit(self, X, y):
        X_ = getData(self.intercept_, X)
            
        self.gd_.fit(X_, y)
        pass
    
    
    def predict(self, X):
        X_ = getData(self.intercept_, X)
        w = self.gd_.get_w()
        
        return 1*( Logistic.logistic(X_, w) > .5)
    
    
    def proba(self, X):
        X_ = getData(self.intercept_, X)
        w = self.gd_.get_w()
        
        return Logistic.logistic(X_, w)
    