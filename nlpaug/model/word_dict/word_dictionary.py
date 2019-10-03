class WordDictionary:
    def __init__(self, cache=True):
        self.cache = cache

    # pylint: disable=R0201
    def train(self, data):
        raise NotImplementedError

    # pylint: disable=R0201
    def predict(self, data):
        raise NotImplementedError

    # pylint: disable=R0201
    def save(self, model_path):
        raise NotImplementedError

    # pylint: disable=R0201
    def read(self, model_path):
        raise NotImplementedError
