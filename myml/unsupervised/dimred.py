'''
Created on 05/11/2013

@author: buhos_000
'''

import numpy as np

import base



class PCA(base.AbstractUnsupervisedMethod):
    
    def __init__(self, n_components = 2):
        self.n_components_ = n_components
    
    def fit(self, X):
        _, s, V = np.linalg.svd(X)
        self.P_ = s * V
        self.d_ = s
    
    def transform(self, X):
        return X.dot(self.P_)[:,:self.n_components_]
    
    

class PPCA(base.AbstractUnsupervisedMethod):
    
    def __init__(self):
        pass
    
    def fit(self, X):
        pass
    
    def transform(self, X):
        pass
    
    