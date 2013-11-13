'''
Created on 13/11/2013

@author: buhos_000
'''

from abc import ABCMeta, abstractmethod


class MapReducer():
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def map(self):
        pass
    
    @abstractmethod
    def reduce(self):
        pass



class MapReduceEngine():
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
    
    if name == "serial":
        return SerialMapReduceEngine()
    else: return SerialMapReduceEngine()