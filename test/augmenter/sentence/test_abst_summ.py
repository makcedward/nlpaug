import unittest
import os
import torch
from dotenv import load_dotenv

import nlpaug.augmenter.sentence as nas
import nlpaug.util.text.tokenizer as text_tokenizer

class TestAbstSummAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.model_paths = [
            'facebook/bart-large-cnn',
            't5-small',
            't5-base',
        ]

        cls.text = """
            The history of natural language processing (NLP) generally started in the 1950s, although work can be 
            found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and 
            Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. 
            The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian 
            sentences into English. The authors claimed that within three or five years, machine translation would
            be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, 
            which found that ten-year-long research had failed to fulfill the expectations, funding for machine 
            translation was dramatically reduced. Little further research in machine translation was conducted 
            until the late 1980s when the first statistical machine translation systems were developed.
        """
        cls.texts = [
            cls.text,
            # https://www.sciencedirect.com/science/article/abs/pii/S0957417418307735
            """
                Summarization, is to reduce the size of the document while preserving the meaning, is one of the 
                most researched areas among the Natural Language Processing (NLP) community. Summarization 
                techniques, on the basis of whether the exact sentences are considered as they appear in the 
                original text or new sentences are generated using natural language processing techniques, are 
                categorized into extractive and abstractive techniques. Extractive summarization has been a very 
                extensively researched topic and has reached to its maturity stage. Now the research has shifted 
                towards the abstractive summarization. The complexities underlying with the natural language text 
                makes abstractive summarization a difficult and a challenging task.
            """
        ]

    def test_batch_size(self):
        # 1 per batch
        aug = nas.AbstSummAug(model_path='t5-small', batch_size=1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size = input size
        aug = nas.AbstSummAug(model_path='t5-small', batch_size=len(self.texts))
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size > input size
        aug = nas.AbstSummAug(model_path='t5-small', batch_size=len(self.texts)+1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # input size > batch size
        aug = nas.AbstSummAug(model_path='t5-small', batch_size=2)
        aug_data = aug.augment(self.texts * 2)
        self.assertEqual(len(aug_data), len(self.texts)*2)

    def test_by_device(self):
        if torch.cuda.is_available():
            self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            aug = nas.AbstSummAug(model_path=model_path, device=device)

            self.empty_input(aug)

            for data in [self.text, self.texts]:
                self.substitute(aug, data)

            if device == 'cpu':
                self.assertTrue(device == aug.model.get_device())
            elif 'cuda' in device:
                self.assertTrue('cuda' in aug.model.get_device())

        self.assertLess(0, len(self.model_paths))

    def empty_input(self, aug):
        text = ''

        augmented_data = aug.augment(text)
        self.assertEqual(len(augmented_data), 0)

        texts = []
        augmented_text = aug.augment(text)
        self.assertEqual(len(augmented_data), 0)

    def substitute(self, aug, data):
        augmented_data = aug.augment(data)

        if isinstance(data, list):
            for d, a in zip(data, augmented_data):
                self.assertLess(len(a.split(' ')), len(d.split(' ')))
                self.assertNotEqual(d, a)
        else:
            augmented_text = augmented_data[0]
            self.assertLess(len(augmented_text.split(' ')), len(data.split(' ')))
            self.assertNotEqual(data, augmented_text)
