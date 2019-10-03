import numpy as np


class WordStatistics:
    def __init__(self, cache=True):
        self.cache = cache

    def train(self, data):
        raise NotImplementedError

    def predict(self, data, top_k):
        raise NotImplementedError

    def save(self, model_path):
        raise NotImplementedError

    def read(self, model_path):
        raise NotImplementedError

    @classmethod
    def choice(cls, x, p, size=1):
        return np.random.choice(len(x), size, p=p)
