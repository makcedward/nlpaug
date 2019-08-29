class LanguageModels:
    def __init__(self, cache=True):
        self.cache = cache

    def predict(self, input_tokens, target_token=None, top_n=5):
        raise NotImplementedError()
