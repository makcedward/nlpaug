try:
    import nltk
    from nltk.corpus import wordnet
except ImportError:
    # No installation required if not using this function
    pass

from nlpaug.model.word_dict import WordDictionary


class WordNet(WordDictionary):
    def __init__(self, lang, is_synonym=True):
        super().__init__(cache=True)

        self.lang = lang
        self.is_synonym = is_synonym

        try:
            import nltk
            from nltk.corpus import wordnet
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed nltk library. Install nltk by `pip install nltk`')

        # try:
        #     # Check whether wordnet package is downloaded
        #     wordnet.synsets('computer')
        #     # Check whether POS package is downloaded
        #     nltk.pos_tag('computer')
        # except LookupError:
        #     nltk.download('wordnet')
        #     nltk.download('averaged_perceptron_tagger')

        self.model = self.read()

    def read(self):
        try:
            wordnet.synsets('testing', lang=self.lang)
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            wordnet.synsets('testing', lang=self.lang)
        return wordnet

    def predict(self, word, pos=None):
        results = []
        for synonym in self.model.synsets(word, pos=pos, lang=self.lang):
            for lemma in synonym.lemmas(lang=self.lang):
                if self.is_synonym:
                    results.append(lemma.name())
                else:
                    for antonym in lemma.antonyms():
                        results.append(antonym.name())
        return results

    @classmethod
    def pos_tag(cls, tokens):
        try:
            results = nltk.pos_tag(tokens)
        except FileNotFoundError:
            # NLTK can leave a resource directory behind after an incomplete
            # download, so force replacement when one of its files is missing.
            nltk.download('averaged_perceptron_tagger_eng', force=True)
            results = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
            results = nltk.pos_tag(tokens)

        return results
