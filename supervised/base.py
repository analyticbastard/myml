
from ABC import ABCMeta, absractmethod

class AbstractSupervisedMethod(object):
    __metaclass__ = ABCMeta
    
    @absractmethod
    def fit(self, X, y):
        pass
    
    @absractmethod
    def predict(self, X):
        pass