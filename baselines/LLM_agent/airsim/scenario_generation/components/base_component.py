import random

class BaseComponent:
    def __init__(self, name):
        self.name = name
    
    def to_dict(self):
        return {'name': self.name}

    def to_vec(self):
        raise NotImplementedError()
    
    @classmethod
    def from_dict(cls, data):
        return cls(data['name'])
    
    @classmethod
    def from_vec(cls, data):
        raise NotImplementedError()
    
    @classmethod
    def crossover(cls, component_1, component_2):
        raise NotImplementedError()
    
    def mutate(self, mutation_rate=0.3):
        raise NotImplementedError()



