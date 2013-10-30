
from abc import ABCMeta, abstractmethod

class AbstractUnsupervisedMethod(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def transform(self, X):
        pass