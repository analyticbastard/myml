'''
Created on 13/11/2013

@author: Javier Arriero-Pais (Analytic Bastard)
'''

from abc import ABCMeta, abstractmethod


class MapReducer():
    """
    MapReduce abstract class which publishes a map and a reduce method.
    Concrete classes that implement this class are the MapReduce programs,
    which implement both mappers and reducers.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def map(self, k1, v1):
        """
        Map method to implement the mapper. The function must return a generator
        by using Python's yield with two values, as in 
            yield k2, v2
        :param k1: Instance key
        :param v1: Instance value
        """
        pass
    
    @abstractmethod
    def reduce(self, k2, v2):
        """
        Reduce method to implement the reducer. The function must return a generator
        by using Python's yield with two values, as in 
            yield k3, v3
        :param k2: Instance key (the mapper's output key)
        :param v2: Instance value (the mapper's output value) 
        """
        pass



class MapReduceEngine():
    """
    Abstract class exposing an interface to the MapReduce framework.
    Concrete classes implementing this interface implement methods to perform
    the MapReduce computations.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__(self):
        pass
    
    def set_mr_class(self, mr):
        self.mr_ = mr
    
    @abstractmethod
    def map(self):
        pass
    
    @abstractmethod
    def reduce(self):
        pass



class SerialMapReduceEngine(MapReduceEngine):
    """
    Concrete class implementing a serial MapReduceEngine. This is a simple
    class that first iteratively performs all mappings and then use their
    output to call the reduce function of its MapReducer.
    
    From the mapper's output generator, this MapReduceEngine generates a
    dictionary for each key and an associated list with the values found.
    Then, for each key, the reduce function is called with the list of
    all its value as argument. 
    """
    def __init__(self, mr = None):
        self.mr_ = mr
        
    def set_datasource(self, data):
        self.datasource_ = data
        
    def start(self):
        redinput = {}
        redoutput = {}
        
        for key, value in self.datasource_:
            gen = self.mr_.map(key, value)
            for newkey, newvalue in gen:
                if redinput.has_key(newkey):
                    redinput[newkey].append(newvalue)
                else:
                    redinput[newkey] = [newvalue]
                    
        for key in redinput:
            values = redinput[key]
            gen = self.mr_.reduce(key, values)
            for newkey, value in gen:
                redoutput[newkey] = value
            
        return redoutput
            


def mapredFactory(name):
    """
    Factory method to isolate MapReduce users from MapReduceEngine
    implementations available, leaving that to the calling program.
    """
    if name == "serial":
        return SerialMapReduceEngine()
    else: return SerialMapReduceEngine()