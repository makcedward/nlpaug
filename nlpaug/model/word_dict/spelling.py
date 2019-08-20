"""
    Source data:
    English Neutral Rewriting: https://github.com/ybisk/charNMT-noise/blob/master/noise/en.natural
"""
from nlpaug.model.word_dict import WordDictionary


class Spelling(WordDictionary):
    def __init__(self, dict_path, include_reverse=True, cache=True):
        super().__init__(cache)

        self.dict_path = dict_path
        self.include_reverse = include_reverse

        self._init()

    def _init(self):
        self.dict = {}
        self.read(self.dict_path)

    def read(self, model_path):
        with open(model_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]

                if key not in self.dict:
                    self.dict[key] = []

                self.dict[key].extend(values)
                # Remove duplicate mapping
                self.dict[key] = list(set(self.dict[key]))
                # Build reverse mapping
                if self.include_reverse:
                    for value in values:
                        if value not in self.dict:
                            self.dict[value] = []
                        if key not in self.dict[value]:
                            self.dict[value].append(key)

    def predict(self, data):
        if data not in self.dict:
            return None

        return self.dict[data]
