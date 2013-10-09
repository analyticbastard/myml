
from abc import ABCMeta, abstractmethod

class AbstractSupervisedMethod(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass