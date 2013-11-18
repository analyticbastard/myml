'''
Created on 06/11/2013

@author: Javier Arriero-Pais (Analytic Bastard)
'''

import numpy as np

import base

import classification

from ..math import mapred



class NaiveBayes(base.AbstractSupervisedMethod):
    """
    Naive Bayes classification implementation with MapReduce.
    """
    
    class NBMapReduce(mapred.MapReducer):
        """
        MapReducer implementation of Naive Bayes classification.
        
        The map and reduce functions output generators to be used by
        MapReduceEngine.
        """
        def map(self, key, value):
            """
            Each mapper yields a key composed of both the label's value
            and the variable X index within the dataset bi-dimensional array
            (its column). The mapper yields the variable categorical value
            as the mapper's output value.
             
            For each variable in multivariate X yield
            (y value, variable index i), k category value
            
            :param key: Input instance from the original dataset X (a row)
            :param value: Label that belongs to this instance
            """
            for i, k in enumerate(key):
                yield (value, i), k
        
        def reduce(self, key, values):
            """
            The reducer receives the conditioning label and the variable column
            as keys, so that it knows the frequency of each category within the
            variable (column). Thus, we can compute the conditioned probability
            P(X_{column} = category | y = label).
            :param key: Tuple (label, column index)
            :param value: category within column in the key
            """
            val = set(values)
            N = len(values)
            # For each variable category, emit
            # (y value, variable index, category value), probability
            for newkey in val:
                yield (key[0], key[1], newkey), 1.0*np.sum(values == newkey)/N
    
    
    def __init__(self):
        """
        Init function where we create the MapReduceEngine using the factory
        method.
        """
        nbmr = NaiveBayes.NBMapReduce()
        self.server_ = mapred.mapredFactory("serial")
        self.server_.set_mr_class(nbmr)
    
    
    def fit(self, X, y):
        """
        Fit data to labels and extract reverse conditioned probabilities
        according to Bayes theorem and Naive Bayes assumptions (variable
        independece).
        :param X: Data
        :param y: labels, zero or one (0, 1)
        """
        datasource = zip(X,y)
        self.server_.set_datasource(datasource)
        self.CP_ = self.server_.start()
        self.PP_ = 1.0*np.array([len(y)-np.sum(y), np.sum(y)])/len(y)
    
    def predict(self, X):
        """
        Predict the labels of the input data according to the probabilities
        stored in the object.
        :param X: Data to classify
        """
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