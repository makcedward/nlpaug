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
        cls.uk_keyboard_path = os.path.join(os.environ.get("PACKAGE_DIR"), 'res', 'char', 'keyboard', 'uk.json')

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

    def test_unicode_letters_are_not_special_characters(self):
        model = nmc.Keyboard(
            model_path=self.uk_keyboard_path,
            special_char=False,
            numeric=False,
            upper_case=False,
        )

        self.assertIn('п', model.model)
        self.assertGreater(len(model.predict('п')), 0)
        self.assertTrue(all(key.isalpha() for key in model.model))
        self.assertTrue(all(value.isalpha() for values in model.model.values() for value in values))

    def test_numeric_filter_uses_isnumeric_digit_boundary(self):
        import json
        import tempfile

        mapping = {
            'a': ['b'],
            '1': ['2'],
            '²': ['³'],
            '!': ['@'],
        }
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf8') as f:
            json.dump(mapping, f)
            path = f.name
        try:
            model = nmc.Keyboard(model_path=path, special_char=True, numeric=False, upper_case=False)
            # '1' and superscript '²' are both isnumeric -> dropped when numeric=False
            self.assertNotIn('1', model.model)
            self.assertNotIn('²', model.model)
            # plain letters and special chars stay (special_char=True)
            self.assertIn('a', model.model)
            self.assertIn('!', model.model)

            digits_only = nmc.Keyboard(model_path=path, special_char=False, numeric=True, upper_case=False)
            # special chars are not alnum -> dropped when special_char=False
            self.assertNotIn('!', digits_only.model)
            self.assertIn('a', digits_only.model)
            self.assertIn('1', digits_only.model)
        finally:
            os.unlink(path)
