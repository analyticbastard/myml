'''
Created on 06/11/2013

@author: buhos_000
'''

import numpy as np

import base

import classification

from ..math import mapred



class NaiveBayes(base.AbstractSupervisedMethod):
    
    class NBMapReduce(mapred.MapReducer):
        
        def map(self, key, value):
            # For each variable in multivariate X yield
            # (y value, variable index i), k category value
            for i, k in enumerate(key):
                yield (value, i), k
        
        def reduce(self, key, values):
            val = set(values)
            N = len(values)
            # For each variable category, emit
            # (y value, variable index, category value), probability
            for newkey in val:
                yield (key[0], key[1], newkey), 1.0*np.sum(values == newkey)/N
    
    
    def __init__(self):
        nbmr = NaiveBayes.NBMapReduce()
        self.server_ = mapred.mapredFactory("serial")
        self.server_.set_mr_class(nbmr)
    
    
    def fit(self, X, y):
        datasource = zip(X,y)
        self.server_.set_datasource(datasource)
        self.CP_ = self.server_.start()
        self.PP_ = 1.0*np.array([len(y)-np.sum(y), np.sum(y)])/len(y)
    
    def predict(self, X):
        N = X.shape[0] if len(X.shape)==2 else 1
        prob = np.zeros((N,1))
        for v in range(N):
            p = 1;
            x = X[v,:] if len(X.shape)==2 else X
            keys = [((0,i,e),(1,i,e)) for i, e in enumerate(x)]
            for k0, k1 in keys:
                try:
                    ptemp0 = self.CP_[k0]
                except:
                    ptemp0 = .000001
                    ptemp1 = 1
                    
                try:
                    ptemp1 = self.CP_[k1]
                except:
                    ptemp1 = .000001
                    ptemp0 = 1
                    
                p *= ptemp0/ptemp1
                p = 1 if p>1 else p
        
            p *= self.PP_[0]/self.PP_[1]
            prob[v] = classification.logistic(-np.log( p ))
            
        return prob