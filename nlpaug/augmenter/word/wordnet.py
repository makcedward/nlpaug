import nltk
from nltk.corpus import wordnet

from nlpaug.augmenter.word import WordAugmenter
from nlpaug.util import Action, PartOfSpeech


class WordNetAug(WordAugmenter):
    def __init__(self, name='WordNet_Aug', aug_min=1, aug_p=0.3, tokenizer=None, stopwords=[], verbose=0):
        super(WordNetAug, self).__init__(
            action=Action.SUBSTITUTE, name=name, aug_p=aug_p, aug_min=aug_min, tokenizer=tokenizer, stopwords=stopwords,
            verbose=verbose)

        self.model = self.get_model()

    def skip_aug(self, token_idxes, pos):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with synonym. It will be excluded in lucky draw.
            if pos[token_idx][1] not in ['DT']:
                results.append(token_idx)

        return results

    def substitute(self, text):
        results = []

        tokens = self.tokenizer(text)

        pos = nltk.pos_tag(tokens)

        aug_idexes = self._get_aug_idxes(tokens)

        for i, token in enumerate(tokens):
            # Skip if no augment for word
            if i not in aug_idexes:
                results.append(token)
                continue

            word_poses = PartOfSpeech.pos2wn(pos[i][1])
            synets = []
            if word_poses is None or len(word_poses) == 0:
                # Use every possible words as the mapping does not defined correctly
                synets.extend(self.model.synsets(pos[i][0]))
            else:
                for word_pos in word_poses:
                    synets.extend(self.model.synsets(pos[i][0], pos=word_pos))

            augmented_data = []
            for synet in synets:
                for candidate in synet.lemma_names():
                    if candidate.lower() != token.lower():
                        augmented_data.append(candidate)

            if len(augmented_data) == 0:
                results.append(token)
            else:
                candidate = self.sample(augmented_data, 1)[0]
                results.append(self.align_capitalization(token, candidate))

        return self.reverse_tokenizer(results)

    def get_model(self):
        try:
            # Check whether wordnet package is downloaded
            wordnet.synsets('computer')
        except Exception:
            nltk.download('wordnet')

        try:
            # Check whether POS package is downloaded
            nltk.pos_tag('computer')
        except Exception:
            nltk.download('averaged_perceptron_tagger')

        return wordnet