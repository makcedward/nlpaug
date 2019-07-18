class WordDictionary:
    def __init__(self, cache=True):
        self.cache = cache

    def train(self, data):
        raise NotImplemented()

    def predict(self, data):
        raise NotImplemented()

    def save(self, model_path):
        raise NotImplemented()

    def read(self, model_path):
        raise NotImplemented()
