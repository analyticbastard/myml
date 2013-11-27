'''
Created on 27/11/2013

@author: buhos_000
'''

from abc import ABCMeta, abstractmethod

import numpy as np

    
def distanceFactory(name, **kwargs):
    if name == "euclidean":
        return DistanceEuclidean()
    else: return DistanceEuclidean()
    
    

class Distance(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def dist(self, X, Y):
        pass


class DistanceEuclidean(Distance):
    
    def __init__(self):
        pass
    
    def dist(self, X, Y):
        vec = X-Y
        sqd = vec.dot(vec)
        return np.sqrt(sqd)
    