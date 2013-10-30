"""
Created on Wed Aug 14 15:26:50 2013

@author: Javier "Analytic Bastard" Arriero-Pais
"""

import sys

import numpy as np

import base



class KMeans(base.AbstractUnsupervisedMethod):
    
    def __init__(self, k = 2, maxiter = 100):
        self.k_ = k
        self.maxiter_ = maxiter
    
    
    def fit(self, X):
        n, self.m_ = X.shape
        
        indices = np.random.randint(0, n, self.k_)
        
        centr = X[:,indices]
        
        # One column for distances, another for centroid index
        assig = np.zeros((n,2))
        assig[:,0] = sys.maxint
        
        for it in range(self.maxiter_):
            assig = self.maximization(X, centr, assig)
            centr = self.expectation(X, centr, assig)
        
        self.centr_ = centr
         
            
    
    def maximization(self,X,centr,assig):
        for j in range(self.k_):
            dist = np.sum((X-centr[j,:]).abs(), axis=1)
            ind  = dist<assig[:,0]
            assig[ind,0] = dist[ind]
            assig[ind,1] = j
            
        return assig
    
    
    def expectation(self,X,centr,assig):
        for j in range(self.k_):
            ind = assig[:,1] == j
            centr[j, :] = np.mean(X[ind,:], axis=0)
    
    
    
    def transform(self, X):
        n, m = X.shape
        if m != self.m_:
            raise Exception
        
        # One column for distances, another for centroid index
        assig = np.zeros((n,2))
        assig[:,0] = sys.maxint
        self.maximization(X, self.centr_, assig)