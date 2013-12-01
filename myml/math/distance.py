'''
Created on 27/11/2013

@author: buhos_000
'''

from abc import ABCMeta, abstractmethod

import numpy as np

    
def distanceFactory(name, **kwargs):
    """
    Factory to create distances based on the name
    """
    if name == "euclidean":
        return DistanceEuclidean()
    else: return DistanceEuclidean()
    
    

class Distance(object):
    """
    Abstract class for a distance
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def dist(self, X, Y):
        pass


class DistanceEuclidean(Distance):
    """
    Class that implements the euclidean distance
    """
    def __init__(self):
        pass
    
    def dist(self, X, Y):
        vec = X-Y
        sqd = vec.dot(vec)
        return np.sqrt(sqd)
    