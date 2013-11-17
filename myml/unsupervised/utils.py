'''
Created on 17/11/2013

@author: buhos_000
'''

import base

class Scale(base.AbstractUnsupervisedMethod):
    
    def __init__(self):
        pass
    
    
    def fit(self, X):
        self.X_ = X
        
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        
        if any(self.std_ == 0):
            raise Exception()
        
        self.transformed_X_ = (X - self.mean_) / self.std_
    
    
    def transform(self, X):
        if X is self.X_:
            return self.transformed_X_
        
        return (X - self.mean_) / self.std_