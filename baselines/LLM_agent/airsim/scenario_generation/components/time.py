from .base_component import BaseComponent 
import random

class Time(BaseComponent):
    def __init__(self, hour, minute, name='time_comp'):
        super().__init__(name)
        self.hour = hour
        self.minute = minute

    def to_dict(self):
        return {'hour': self.hour, 'minute': self.minute}
    
    def to_vec(self):
        return [self.hour, self.minute]
    

    @classmethod
    def from_dict(cls, data):
        return cls(data['hour'], data['minute'])
    
    @classmethod    
    def from_vec(cls, data):
        return cls(*data)
    
    @classmethod
    def crossover(cls, time_1, time_2):
        return cls(time_1.hour, time_2.minute), cls(time_2.hour, time_1.minute)
    
    def mutate(self, mutation_rate=0.3):
        flag = random.random()
        if flag <= mutation_rate:
            self.hour += random.uniform(-0.1, 0.1)
            self.minute += random.uniform(-0.1, 0.1)
            self.hour = max(0, min(1, self.hour))
            self.minute = max(0, min(1, self.minute))