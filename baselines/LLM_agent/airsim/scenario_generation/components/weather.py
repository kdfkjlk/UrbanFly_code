
from .base_component import BaseComponent
import random

class Weather(BaseComponent):
    def __init__(self, rain, road_wetness, snow, road_snow, maple_leaf, road_leaf, dust, fog, wind, name='weather_comp'):
        super().__init__(name)
        self.rain = rain
        self.road_wetness = road_wetness
        self.snow = snow
        self.road_snow = road_snow
        self.maple_leaf = maple_leaf
        self.road_leaf = road_leaf
        self.dust = dust
        self.fog = fog
        self.wind = wind

    def to_vec(self):
        return [self.rain, self.road_wetness, self.snow, self.road_snow, self.maple_leaf,
         self.road_leaf, self.dust, self.fog, self.wind]
        
    def to_dict(self):
        return {
            'rain': self.rain,
            'road_wetness': self.road_wetness,
            'snow': self.snow,
            'road_snow': self.road_snow,
            'maple_leaf': self.maple_leaf,
            'road_leaf': self.road_leaf,
            'dust': self.dust,
            'fog': self.fog,
            'wind': self.wind
        }


    @classmethod
    def from_dict(cls, data):
        return cls(data['rain'], data['road_wetness'], data['snow'], data['road_snow'], 
                   data['maple_leaf'], data['road_leaf'], data['dust'], data['fog'], data['wind'])
    
    @classmethod
    def from_vec(cls, data):
        return cls(*data)  
    
    @classmethod
    def crossover(cls, weather_1, weather_2):
        weather_1_vec = weather_1.to_vec()
        weather_2_vec = weather_2.to_vec()
        half_len = len(weather_1_vec) // 2
        new_weather_vec_1 = weather_1_vec[:half_len] + weather_2_vec[half_len:]
        new_weather_vec_2 = weather_2_vec[:half_len] + weather_1_vec[half_len:]

        return cls.from_vec(new_weather_vec_1), cls.from_vec(new_weather_vec_2)
    
    def mutate(self, mutation_rate=0.3):
        flag = random.random()
        if flag <= mutation_rate:
            self.rain += random.uniform(-0.1, 0.1)
            self.road_wetness += random.uniform(-0.1, 0.1)
            self.snow += random.uniform(-0.1, 0.1)
            self.road_snow += random.uniform(-0.1, 0.1)
            self.maple_leaf += random.uniform(-0.1, 0.1)
            self.road_leaf += random.uniform(-0.1, 0.1)
            self.dust += random.uniform(-0.1, 0.1)
            self.fog += random.uniform(-0.1, 0.1)
            self.wind += random.uniform(-0.1, 0.1)

            self.rain = max(min(self.rain, 0.5), 0)
            self.road_wetness = max(min(self.road_wetness, 0.5), 0)
            self.snow = max(min(self.snow, 0.5), 0)
            self.road_snow = max(min(self.road_snow, 0.5), 0)
            self.maple_leaf = max(min(self.maple_leaf, 0.5), 0)
            self.road_leaf = max(min(self.road_leaf, 0.5), 0)
            self.dust = max(min(self.dust, 0.5), 0)
            self.fog = max(min(self.fog, 0.5), 0)
            self.wind = max(min(self.wind, 0.5), 0)