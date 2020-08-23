import unittest
import re
import os
from dotenv import load_dotenv

import nlpaug.model.char as nmc


class TestKeyboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.eng_keyboard_path = os.path.join(os.environ.get("PACKAGE_DIR"), 'res', 'char', 'keyboard', 'en.json')

    def test_lower_case_only(self):
        model = nmc.Keyboard(model_path=self.eng_keyboard_path, special_char=False, numeric=False, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-z]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-z]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_special_char_lower_case(self):
        model = nmc.Keyboard(model_path=self.eng_keyboard_path, special_char=True, numeric=False, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertFalse(re.match("^[0-9]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertFalse(re.match("^[0-9]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_numeric_lower_case(self):
        model = nmc.Keyboard(model_path=self.eng_keyboard_path, special_char=False, numeric=True, upper_case=False)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-z0-9]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-z0-9]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_upper_lower_case(self):
        model = nmc.Keyboard(model_path=self.eng_keyboard_path, special_char=False, numeric=False, upper_case=True)
        mapping = model.model
        for key, values in mapping.items():
            self.assertTrue(re.match("^[a-zA-Z]*$", key))
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertTrue(re.match("^[a-zA-Z]*$", value))
        self.assertGreater(len(mapping), 0)

    def test_special_char_numeric_lower_case(self):
        model = nmc.Keyboard(model_path=self.eng_keyboard_path, special_char=True, numeric=True, upper_case=True)
        mapping = model.model
        for key, values in mapping.items():
            self.assertGreater(len(values), 0)
        self.assertGreater(len(mapping), 0)
