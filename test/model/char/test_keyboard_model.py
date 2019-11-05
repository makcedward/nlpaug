import unittest
import re

import nlpaug.model.char as nmc


class TestKeyboard(unittest.TestCase):
    def test_lower_case_only(self):
        model = nmc.Keyboard(special_char=False, numeric=False, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-z]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-z]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_special_char_lower_case(self):
        model = nmc.Keyboard(special_char=True, numeric=False, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertFalse(re.match("^[0-9]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertFalse(re.match("^[0-9]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_numeric_lower_case(self):
        model = nmc.Keyboard(special_char=False, numeric=True, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-z0-9]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-z0-9]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_upper_lower_case(self):
        model = nmc.Keyboard(special_char=False, numeric=False, upper_case=True)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-zA-Z]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-zA-Z]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_special_char_numeric_lower_case(self):
        model = nmc.Keyboard(special_char=True, numeric=True, upper_case=True)
        mapping = model.model
        for key, values in mapping.items():
            self.assertGreater(len(values), 0)
        self.assertGreater(len(mapping), 0)
