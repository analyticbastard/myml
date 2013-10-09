
import numpy as np

import scipy as sp

import base


class NNLS(base.AbstractSupervisedMethod):
    
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
            
            z = sp.linalg.lstsq(Bs, y)
            print z
            z[self.E] = 0
            
            while np.any( z[self.S] <= 0 ):
                alpha = np.min( g/(g-z) )
                g = g + alpha * ( z - g )
                
                self.move_S_E( g )
                
                Bs = np.zeros( (n, m) )
                Bs[:,self.S] = X[:, self.S ]
                
                z = sp.linalg.lstsq(Bs, y)
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
            w = b.T.dot ( y - X.dot(g) )
            
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