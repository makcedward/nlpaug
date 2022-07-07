import unittest
import os
from dotenv import load_dotenv
import torch

import nlpaug.augmenter.word as naw
import nlpaug.model.lang_models as nml


class TestContextualWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.text = 'The quick brown fox jumps over the lazy dog. '
        cls.texts = [
            'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.',
            "Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece that wasn't quite meant to be."
        ]
        cls.debug = False

        cls.model_paths = [
            'distilbert-base-uncased',
            'bert-base-uncased',
            'bert-base-cased',
            'roberta-base',
            'distilroberta-base',
            'allenai/longformer-base-4096',
            'squeezebert/squeezebert-uncased',
        ]

    def test_quicktest(self):
        for model_path in self.model_paths:
            if self.debug:
                print('=============:', model_path)
            aug = naw.ContextualWordEmbsAug(model_path=model_path)
            text = 'The quick brown fox jumps over the lazaaaaaaaaay dog'
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            # print('[{}]: {}'.format(model_path, augmented_text))
            self.assertNotEqual(text, augmented_text)

    def test_incorrect_model_name(self):
        with self.assertRaises(ValueError) as error:
            naw.ContextualWordEmbsAug(model_path='unknown')

        self.assertTrue('Model type value is unexpected.' in str(error.exception))

    def test_none_device(self):
        for model_path in self.model_paths:
            aug = naw.ContextualWordEmbsAug(
                model_path=model_path, force_reload=True, device=None)
            self.assertTrue(aug.device == 'cpu')

    def test_reset_model(self):
        for model_path in self.model_paths:
            original_aug = naw.ContextualWordEmbsAug(
                    model_path=model_path, action="insert", force_reload=True)
            original_top_k = original_aug.model.top_k

            new_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="insert", force_reload=True,
                top_k=original_top_k+1)
            new_top_k = new_aug.model.top_k

            self.assertEqual(original_top_k + 1, new_top_k)

    def test_multilingual(self):
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased')

        inputs = [
            {'lang': 'fra', 'text': "Bonjour, J'aimerais une attestation de l'employeur certifiant que je suis en CDI."},
            {'lang': 'jap', 'text': '速い茶色の狐が怠惰なな犬を飛び越えます'},
            {'lang': 'spa', 'text': 'un rapido lobo marron salta sobre el perro perezoso'}
        ]

        for input_param in inputs:
            augmented_data = aug.augment(input_param['text'])
            augmented_text = augmented_data[0]
            self.assertNotEqual(input_param['text'], augmented_text)
            # print('[{}]: {}'.format(input_param['lang'], augmented_text))

    def test_fast_tokenizer(self):
        aug = naw.ContextualWordEmbsAug(model_path="blinoff/roberta-base-russian-v0", force_reload=True)
        aug.augment("Мозг — это машина  которая пытается снизить ошибку в прогнозе.")
        self.assertTrue(True)

    def test_model_type(self):
        aug = naw.ContextualWordEmbsAug(model_path="blinoff/roberta-base-russian-v0", model_type='roberta', force_reload=True)
        aug.augment("Мозг — это машина  которая пытается снизить ошибку в прогнозе.")
        self.assertTrue(True)

    def test_batch_size(self):
        # 1 per batch
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', model_type='bert', batch_size=1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size = input size
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', model_type='bert', 
            batch_size=len(self.texts))
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # batch size > input size
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', model_type='bert', 
            batch_size=len(self.texts)+1)
        aug_data = aug.augment(self.texts)
        self.assertEqual(len(aug_data), len(self.texts))

        # input size > batch size
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', model_type='bert', 
            batch_size=2)
        aug_data = aug.augment(self.texts * 2)
        self.assertEqual(len(aug_data), len(self.texts)*2)

    def test_contextual_word_embs(self):
        if torch.cuda.is_available():
            self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            if self.debug:
                print('=============:', model_path)
            insert_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="insert", force_reload=True, device=device)
            substitute_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="substitute", device=device)

            if device == 'cpu':
                self.assertTrue(device == insert_aug.model.get_device())
                self.assertTrue(device == substitute_aug.model.get_device())
            elif 'cuda' in device:
                self.assertTrue('cuda' in insert_aug.model.get_device())
                self.assertTrue('cuda' in substitute_aug.model.get_device())

            for data in [self.text, self.texts]:
                self.insert(insert_aug, data)
                self.substitute(substitute_aug, data)
                if self.debug:
                    print('=============data:', data)
                self.substitute_stopwords(substitute_aug, data)
                self.decode_by_tokenizer([insert_aug, substitute_aug])

            self.subword([insert_aug, substitute_aug])
            self.max_length([insert_aug, substitute_aug])
            self.empty_replacement(substitute_aug)
            self.skip_short_token(substitute_aug)
            
        self.assertLess(0, len(self.model_paths))

    def skip_short_token(self, aug):
        text = 'I am a boy'
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertNotEqual(text.lower(), augmented_text.lower())

        original_aug_min = aug.aug_min
        aug.aug_min = 4
        augmented_data = aug.augment(text)
        augmented_text = augmented_data[0]
        self.assertEqual(text.lower(), augmented_text.lower())
        aug.aug_min = original_aug_min

    def decode_by_tokenizer(self, augs):
        text = "I don't get it actually"
        for aug in augs:
            original_aug_min = aug.aug_min
            aug.aug_min = 4
            augmented_data = aug.augment(text)
            augmented_text = augmented_data[0]
            self.assertTrue("'t" in augmented_text and " 't'" not in augmented_text)
            aug.aug_min = original_aug_min

    def insert(self, aug, data):
        self.assertLess(0, len(data))
        augmented_data = aug.augment(data)

        if isinstance(data, list):
            for d, a in zip(data, augmented_data):
                self.assertNotEqual(d, a)
                self.assertTrue(aug.model.get_subword_prefix() not in a)
        else:
            augmented_text = augmented_data[0]
            self.assertNotEqual(data, augmented_text)
            self.assertTrue(aug.model.get_subword_prefix() not in augmented_text)

    def substitute(self, aug, data):
        augmented_data = aug.augment(data)

        if isinstance(data, list):
            for d, a in zip(data, augmented_data):
                self.assertNotEqual(d, a)
                self.assertTrue(aug.model.get_subword_prefix() not in a)
        else:
            augmented_text = augmented_data[0]
            self.assertNotEqual(data, augmented_text)
            self.assertTrue(aug.model.get_subword_prefix() not in augmented_text)

    def substitute_stopwords(self, aug, data):
        original_stopwords = aug.stopwords
        if isinstance(data, list):
            stopwords = [t.lower() for t in data[0].split(' ')[:3]]
            aug.stopwords = stopwords
            aug._build_stop_words(stopwords)
        else:
            stopwords = [t.lower() for t in data.split(' ')[:3]]
            aug.stopwords = stopwords
            aug._build_stop_words(stopwords)
        aug_n = 3

        self.assertLess(0, len(data))

        try_cnt = 5
        for _ in range(try_cnt):
            augmented_cnt = 0
            augmented_data = aug.augment(data)

            if isinstance(data, list):
                for d, augmented_text in zip(data, augmented_data):
                    augmented_tokens = aug.tokenizer(augmented_text)
                    tokens = aug.tokenizer(d)
                    for token, augmented_token in zip(tokens, augmented_tokens):
                        if token.lower() in aug.stopwords and len(token) > aug_n:
                            self.assertEqual(token.lower(), augmented_token)
                        else:
                            augmented_cnt += 1

                    self.assertGreater(augmented_cnt, 3)
            else:
                augmented_text = augmented_data[0]
                augmented_tokens = aug.tokenizer(augmented_text)
                tokens = aug.tokenizer(data)

                for token, augmented_token in zip(tokens, augmented_tokens):
                    if token.lower() in aug.stopwords and len(token) > aug_n:
                        self.assertEqual(token.lower(), augmented_token)
                    else:
                        augmented_cnt += 1

                self.assertGreater(augmented_cnt, 3)

        aug.stopwords = original_stopwords
        aug._build_stop_words(original_stopwords)

    def subword(self, augs):
        # https://github.com/makcedward/nlpaug/issues/38
        text = "If I enroll in the ESPP, when will my offering begin and the price set?"
        texts = [self.text, text]

        for _ in range(10):
            for aug in augs:
                aug.augment(text)
                aug.augment(texts)

        self.assertTrue(True)

    def max_length(self, augs):
        # from IMDB v1
        text = """
            Seeing all of the negative reviews for this movie, I figured that it could be yet another comic masterpiece
            that wasn't quite meant to be. I watched the first two fight scenes, listening to the generic dialogue
            delivered awfully by Lungren, and all of the other thrown-in Oriental actors, and I found the movie so
            awful that it was funny. Then Brandon Lee enters the story and the one-liners start flying, the plot falls
            apart, the script writers start drinking and the movie wears out it's welcome, as it turns into the worst
            action movie EVER.<br /><br />Lungren beats out his previous efforts in "The Punisher" and others, as well
            as all of Van Damme's movies, Seagal's movies, and Stallone's non-Rocky movies, for this distinct honor.
            This movie has the absolute worst acting (check out Tia Carrere's face when she is in any scene with Dolph,
            that's worth a laugh), with the worst dialogue ever (Brandon Lee's comment about little Dolph is the worst
            line ever in a film), and the worst outfit in a film (Dolph in full Japanese attire). Picture "Tango and
            Cash" with worse acting, meets "Commando," meets "Friday the 13th" (because of the senseless nudity and
            Lungren's performance is very Jason Voorhees-like), in an hour and fifteen minute joke of a movie.<br />
            <br />The good (how about not awful) performances go to the bad guy (who still looks constipated through
            his entire performance) and Carrere (who somehow says 5 lines without breaking out laughing).
            Brandon Lee is just there being Lungren's sidekick, and doing a really awful job at that.<br /><br />An
            awful, awful movie. Fear it and avoid it. If you do watch it though, ask yourself why the underwater shots
            are twice as clear as most non-underwater shots. Speaking of the underwater shots, check out the lame water
            fight scene with the worst fight-scene-ending ever. This movie has every version of a bad fight scene for
            those with short attention spans and to fill-in between the flashes of nudity.<br /><br />A BAD BAD
            MOVIE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

        texts = [self.text, text]

        for aug in augs:
            augmented_data = aug.augment(texts)
            for augmented_text, orig_text in zip(augmented_data, texts):
                self.assertNotEqual(orig_text, augmented_text)

    # https://github.com/makcedward/nlpaug/pull/51
    def empty_replacement(self, aug):
        text = '"Does what it says on the tin! No messing about, quick, easy and exactly as promised. ' \
               'Couldn\'t fault them."'

        texts = [self.text, text]

        augmented_data = aug.augment(text)
        self.assertNotEqual(text, augmented_data)

        augmented_data = aug.augment(texts)
        for augmented_text, orig_text in zip(augmented_data, texts):
            self.assertNotEqual(orig_text, augmented_text)
