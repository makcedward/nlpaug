class PartOfSpeech:
    NOUN = 'noun'
    VERB = 'verb'
    ADJECTIVE = 'adjective'
    ADVERB = 'adverb'

    pos2con = {
        'n': [
            'NN', 'NNS', 'NNP', 'NNPS',  # from WordNet
            'NP'  # from PPDB
        ],
        'v': [
            'VB', 'VBD', 'VBG', 'VBN', 'VBZ',  # from WordNet
            'VBP'  # from PPDB
        ],
        'a': ['JJ', 'JJR', 'JJS', 'IN'],
        's': ['JJ', 'JJR', 'JJS', 'IN'],  # Adjective Satellite
        'r': ['RB', 'RBR', 'RBS'],  # Adverb
    }

    con2pos = {}
    poses = []
    for key, values in pos2con.items():
        poses.extend(values)
        for value in values:
            if value not in con2pos:
                con2pos[value] = []
            con2pos[value].append(key)

    @staticmethod
    def pos2constituent(pos):
        if pos in PartOfSpeech.pos2con:
            return PartOfSpeech.pos2con[pos]
        return []

    @staticmethod
    def constituent2pos(con):
        if con in PartOfSpeech.con2pos:
            return PartOfSpeech.con2pos[con]
        return []

    @staticmethod
    def get_pos():
        return PartOfSpeech.poses
