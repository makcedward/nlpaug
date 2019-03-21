class PartOfSpeech:
    NOUN = 'noun'
    VERB = 'verb'
    ADJECTIVE = 'adjective'
    ADVERB = 'adverb'

    @staticmethod
    def wn2pos(wn_pos):
        if wn_pos == 'n':
            # Noun
            return ['NN', 'NNS', 'NNP', 'NNPS']
        if wn_pos == 'v':
            # Verb
            return ['VB', 'VBD', 'VBG', 'VBN', 'VBZ']

        if wn_pos in ['a', 's']:
            # Adjective/ Adjective Satellite
            return ['JJ', 'JJR', 'JJS', 'IN']

        if wn_pos == 'r':
            # Adverb
            return ['RB', 'RBR', 'RBS']

    @staticmethod
    def pos2wn(pos):
        if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
            # Noun
            return ['n']
        elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ']:
            # Verb
            return ['v']
        elif pos in ['JJ', 'JJR', 'JJS', 'IN']:
            # Adjective/ Adjective Satellite
            return ['a', 's']
        elif pos in ['RB', 'RBR', 'RBS']:
            # Adverb
            return 'r'

    @staticmethod
    def get_wn_pos():
        return [
            'NN', 'NNS', 'NNP', 'NNPS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBZ',
            'JJ', 'JJR', 'JJS', 'IN',
            'RB', 'RBR', 'RBS'
        ]
