import unittest
import os
from dotenv import load_dotenv

import nlpaug.augmenter.word as naw
import nlpaug.model.lang_models as nml


class TestContextualWordEmbsAug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '.env'))
        load_dotenv(env_config_path)

        cls.text = 'The quick brown fox jumps over the lazy dog'

        cls.model_paths = [
            'distilbert-base-uncased',
            'bert-base-uncased',
            'bert-base-cased',
            'xlnet-base-cased',
            'roberta-base',
            'distilroberta-base'
        ]

    def test_quicktest(self):
        for model_path in self.model_paths:
            aug = naw.ContextualWordEmbsAug(model_path=model_path)
            text = 'The quick brown fox jumps over the lazaaaaaaaaay dog'
            augmented_text = aug.augment(text)
            # print('[{}]: {}'.format(model_path, augmented_text))
            self.assertNotEqual(text, augmented_text)

    def test_incorrect_model_name(self):
        with self.assertRaises(ValueError) as error:
            naw.ContextualWordEmbsAug(model_path='unknown')

        self.assertTrue('Model name value is unexpected.' in str(error.exception))

    def test_none_device(self):
        for model_path in self.model_paths:
            aug = naw.ContextualWordEmbsAug(
                model_path=model_path, force_reload=True, device=None)
            self.assertTrue(aug.device == 'cuda' or aug.device == 'cpu')

    def test_reset_model(self):
        for model_path in self.model_paths:
            original_aug = naw.ContextualWordEmbsAug(
                    model_path=model_path, action="insert", force_reload=True, top_p=0.5)
            original_temperature = original_aug.model.temperature
            original_top_k = original_aug.model.top_k
            original_top_p = original_aug.model.top_p

            new_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="insert", force_reload=True,
                temperature=original_temperature+1, top_k=original_top_k+1, top_p=original_top_p+1)
            new_temperature = new_aug.model.temperature
            new_top_k = new_aug.model.top_k
            new_top_p = new_aug.model.top_p

            self.assertEqual(original_temperature+1, new_temperature)
            self.assertEqual(original_top_k + 1, new_top_k)
            self.assertEqual(original_top_p + 1, new_top_p)

    def test_multilingual(self):
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-multilingual-uncased')

        inputs = [
            {'lang': 'fra', 'text': "Bonjour, J'aimerais une attestation de l'employeur certifiant que je suis en CDI."},
            {'lang': 'jap', 'text': '速い茶色の狐が怠惰なな犬を飛び越えます'},
            {'lang': 'spa', 'text': 'un rapido lobo marron salta sobre el perro perezoso'}
        ]

        for input_param in inputs:
            augmented_text = aug.augment(input_param['text'])
            self.assertNotEqual(input_param['text'], augmented_text)
            # print('[{}]: {}'.format(input_param['lang'], augmented_text))

    def test_contextual_word_embs(self):
        self.execute_by_device('cuda')
        self.execute_by_device('cpu')

    def execute_by_device(self, device):
        for model_path in self.model_paths:
            insert_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="insert", force_reload=True, device=device)
            substitute_aug = naw.ContextualWordEmbsAug(
                model_path=model_path, action="substitute", force_reload=True, device=device)

            self.insert(insert_aug)
            self.substitute(substitute_aug)
            self.substitute_stopwords(substitute_aug)
            self.subword([insert_aug, substitute_aug])
            self.top_k([insert_aug, substitute_aug])
            self.top_p([insert_aug, substitute_aug])
            self.top_k_top_p([insert_aug, substitute_aug])
            self.no_top_k_top_p([insert_aug, substitute_aug])
            self.max_length([insert_aug, substitute_aug])
            self.empty_replacement(substitute_aug)

        self.assertLess(0, len(self.model_paths))

    def insert(self, aug):
        self.assertLess(0, len(self.text))
        augmented_text = aug.augment(self.text)

        self.assertNotEqual(self.text, augmented_text)
        self.assertTrue(nml.Bert.SUBWORD_PREFIX not in augmented_text)

    def substitute(self, aug):
        augmented_text = aug.augment(self.text)

        self.assertNotEqual(self.text, augmented_text)
        self.assertTrue(nml.Bert.SUBWORD_PREFIX not in augmented_text)

    def substitute_stopwords(self, aug):
        original_stopwords = aug.stopwords
        aug.stopwords = [t.lower() for t in self.text.split(' ')[:3]]
        aug_n = 3

        for _ in range(20):
            augmented_cnt = 0
            self.assertLess(0, len(self.text))

            augmented_text = aug.augment(self.text)
            augmented_tokens = aug.tokenizer(augmented_text)
            tokens = aug.tokenizer(self.text)

            for token, augmented_token in zip(tokens, augmented_tokens):
                if token.lower() in aug.stopwords and len(token) > aug_n:
                    self.assertEqual(token.lower(), augmented_token)
                else:
                    augmented_cnt += 1

            self.assertGreater(augmented_cnt, 0)

        aug.stopwords = original_stopwords

    def subword(self, augs):
        # https://github.com/makcedward/nlpaug/issues/38
        text = "If I enroll in the ESPP, when will my offering begin and the price set?"

        for _ in range(100):
            for aug in augs:
                aug.augment(text)

        self.assertTrue(True)

    def top_k(self, augs):
        for aug in augs:
            original_top_k = aug.model.top_k
            original_top_p = aug.model.top_p

            aug.model.top_k = 10000
            aug.model.top_p = None

            augmented_text = aug.augment(self.text)

            self.assertNotEqual(self.text, augmented_text)

            if aug.model_type not in ['roberta']:
                self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

            aug.model.top_k = original_top_k
            aug.model.top_p = original_top_p

    def top_p(self, augs):
        for aug in augs:
            original_top_k = aug.model.top_k
            original_top_p = aug.model.top_p

            aug.model.top_k = None
            aug.model.top_p = 0.5

            at_least_one_not_equal = False
            for _ in range(10):
                augmented_text = aug.augment(self.text)

                if self.text != augmented_text:
                    at_least_one_not_equal = True
                    break

            self.assertTrue(at_least_one_not_equal)

            if aug.model_type not in ['roberta']:
                self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

            aug.model.top_k = original_top_k
            aug.model.top_p = original_top_p

    def top_k_top_p(self, augs):
        for aug in augs:
            original_top_k = aug.model.top_k
            original_top_p = aug.model.top_p

            aug.model.top_k = 1000
            aug.model.top_p = 0.5

            augmented_text = aug.augment(self.text)

            self.assertNotEqual(self.text, augmented_text)

            if aug.model_type not in ['roberta']:
                self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

            aug.model.top_k = original_top_k
            aug.model.top_p = original_top_p

    def no_top_k_top_p(self, augs):
        for aug in augs:
            original_top_k = aug.model.top_k
            original_top_p = aug.model.top_p

            aug.model.top_k = None
            aug.model.top_p = None

            augmented_text = aug.augment(self.text)

            self.assertNotEqual(self.text, augmented_text)

            if aug.model_type not in ['roberta']:
                self.assertTrue(aug.model.SUBWORD_PREFIX not in augmented_text)

            aug.model.top_k = original_top_k
            aug.model.top_p = original_top_p

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
            his entire performance) and Carrere (who somehow says [MASK] 5 lines without breaking out laughing).
            Brandon Lee is just there being Lungren's sidekick, and doing a really awful job at that.<br /><br />An
            awful, awful movie. Fear it and avoid it. If you do watch it though, ask yourself why the underwater shots
            are twice as clear as most non-underwater shots. Speaking of the underwater shots, check out the lame water
            fight scene with the worst fight-scene-ending ever. This movie has every version of a bad fight scene for
            those with short attention spans and to fill-in between the flashes of nudity.<br /><br />A BAD BAD
            MOVIE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

        for aug in augs:
            augmented_text = aug.augment(text)
            self.assertNotEqual(text, augmented_text)

    # https://github.com/makcedward/nlpaug/pull/51
    def empty_replacement(self, aug):
        text = '"Does what it says on the tin! No messing about, quick, easy and exactly as promised. ' \
               'Couldn\'t fault them."'

        augmented_text = aug.augment(text)
        self.assertNotEqual(text, augmented_text)
