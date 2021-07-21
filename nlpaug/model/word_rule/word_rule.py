from nlpaug.model.base_model import Model

class WordRule(Model):
    def __init__(self, cache=True):
        self.cache = cache

    # pylint: disable=R0201
    def predict(self, data):
        raise NotImplementedError
