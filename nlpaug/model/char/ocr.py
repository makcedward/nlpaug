from nlpaug.model.char import Character


class Ocr(Character):
    def __init__(self, cache=True):
        super().__init__(cache)

        self.model = self.get_model()

    def predict(self, data):
        return self.model[data]

    # TODO: Read from file
    @classmethod
    def get_model(cls):
        mapping = {
            '0': ['8', '9', 'o', 'O', 'D'],
            '1': ['4', '7', 'l', 'I'],
            '2': ['z', 'Z'],
            '5': ['8'],
            '6': ['b'],
            '8': ['s', 'S', '@', '&'],
            '9': ['g'],
            'o': ['u'],
            'r': ['k'],
            'C': ['G'],
            'O': ['D', 'U'],
            'E': ['B']
        }

        result = {}

        for k in mapping:
            result[k] = mapping[k]

        for k in mapping:
            for v in mapping[k]:
                if v not in result:
                    result[v] = []

                if k not in result[v]:
                    result[v].append(k)

        return result
