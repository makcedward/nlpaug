import re

from nlpaug.model.char import Character


class Keyboard(Character):
    def __init__(self, special_char=True, numeric=True, upper_case=True, cache=True):
        super().__init__(cache)

        self.special_char = special_char
        self.numeric = numeric
        self.upper_case = upper_case
        self.model = self.get_model(special_char=special_char, numeric=numeric, upper_case=upper_case)

    def predict(self, data):
        return self.model[data]

    # TODO: Read from file and extending to 2 keyboard distance
    @classmethod
    def get_model(cls, special_char=True, numeric=True, upper_case=True):
        mapping = {
            '1': ['!', '2', '@', 'q', 'w'],
            '2': ['@', '1', '!', '3', '#', 'q', 'w', 'e'],
            '3': ['#', '2', '@', '4', '$', 'w', 'e'],
            '4': ['$', '3', '#', '5', '%', 'e', 'r'],
            '5': ['%', '4', '$', '6', '^', 'r', 't', 'y'],
            '6': ['^', '5', '%', '7', '&', 't', 'y', 'u'],
            '7': ['&', '6', '^', '8', '*', 'y', 'u', 'i'],
            '8': ['*', '7', '&', '9', '(', 'u', 'i', 'o'],
            '9': ['(', '8', '*', '0', ')', 'i', 'o', 'p'],

            'q': ['1', '!', '2', '@', 'w', 'a', 's'],
            'w': ['1', '!', '2', '@', '3', '#', 'q', 'e', 'a', 's', 'd'],
            'e': ['2', '@', '3', '#', '4', '$', 'w', 'r', 's', 'd', 'f'],
            'r': ['3', '#', '4', '$', '5', '%', 'e', 't', 'd', 'f', 'g'],
            't': ['4', '$', '5', '%', '6', '^', 'r', 'y', 'f', 'g', 'h'],
            'y': ['5', '%', '6', '^', '7', '&', 't', 'u', 'g', 'h', 'j'],
            'u': ['6', '^', '7', '&', '8', '*', ' t', 'i', 'h', 'j', 'k'],
            'i': ['7', '&', '8', '*', '9', '(', 'u', 'o', 'j', 'k', 'l'],
            'o': ['8', '*', '9', '(', '0', ')', 'i', 'p', 'k', 'l'],
            'p': ['9', '(', '0', ')', 'o', 'l'],

            'a': ['q', 'w', 'a', 's', 'z', 'x'],
            's': ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'],
            'd': ['w', 'e', 'r', 's', 'f', 'x', 'c', 'v'],
            'f': ['e', 'r', 't', 'd', 'g', 'c', 'v', 'b'],
            'g': ['r', 't', 'y', 'f', 'h', 'v', 'b', 'n'],
            'h': ['t', 'y', 'u', 'g', 'j', 'b', 'n', 'm'],
            'j': ['y', 'u', 'i', 'h', 'k', 'n', 'm', ',', '<'],
            'k': ['u', 'i', 'o', 'j', 'l', 'm', ',', '<', '.', '>'],
            'l': ['i', 'o', 'p', 'k', ';', ':', ',', '<', '.', '>', '/', '?'],

            'z': ['a', 's', 'x'],
            'x': ['a', 's', 'd', 'z', 'c'],
            'c': ['s', 'd', 'f', 'x', 'v'],
            'v': ['d', 'f', 'g', 'c', 'b'],
            'b': ['f', 'g', 'h', 'v', 'n'],
            'n': ['g', 'h', 'j', 'b', 'm'],
            'm': ['h', 'j', 'k', 'n', ',', '<']
        }

        result = {}

        for key, values in mapping.items():
            # Skip records if key is numeric while include_numeric is false
            if not numeric and re.match("^[0-9]*$", key):
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
