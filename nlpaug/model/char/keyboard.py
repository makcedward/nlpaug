import re
import os
import json

from nlpaug.model.char import Character


class Keyboard(Character):
    def __init__(self, special_char=True, numeric=True, upper_case=True, cache=True, lang="en", model_path=None):
        super().__init__(cache)

        self.special_char = special_char
        self.numeric = numeric
        self.upper_case = upper_case
        self.lang = lang
        self.model_path = model_path
        self.model = self.get_model(model_path=model_path, special_char=special_char, numeric=numeric, 
            upper_case=upper_case, lang=lang)

    def predict(self, data):
        return self.model[data]

    # TODO: Extending to 2 keyboard distance
    @classmethod
    def get_model(cls, model_path, special_char=True, numeric=True, upper_case=True, lang="en"):
        if not os.path.exists(model_path):
            raise ValueError('The model_path does not exist. Please check "{}"'.format(model_path))

        with open(model_path, encoding="utf8") as f:
            mapping = json.load(f)

        result = {}

        for key, values in mapping.items():
            # Skip records if key is numeric while include_numeric is false
            if not numeric and re.match("^[0-9]*$", key):
                continue
            # skip record if key is special character while include_spec is false
            if not special_char and not re.match("^[a-z0-9]*$", key):
                continue

            result[key] = []
            result[key.upper()] = []

            for value in values:
                # Skip record if value is numeric while include_numeric is false
                if not numeric and re.match("^[0-9]*$", value):
                    continue

                # skip record if value is special character while include_spec is false
                if not special_char and not re.match("^[a-z0-9]*$", value):
                    continue

                result[key].append(value)

                if upper_case:
                    result[key].append(value.upper())
                    result[key.upper()].append(value)
                    result[key.upper()].append(value.upper())

        clean_result = {}
        for key, values in result.items():
            # clear empty mapping
            if len(values) == 0:
                continue

            # de-duplicate
            values = [v for v in values if v != key]
            values = sorted(list(set(values)))

            clean_result[key] = values

        return clean_result
