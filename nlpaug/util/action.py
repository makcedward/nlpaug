class Action:
    INSERT = 'insert'
    SUBSTITUTE = 'substitute'
    DELETE = 'delete'
    SWAP = 'swap'
    SPLIT = 'split'

    SEQUENTIAL = 'sequential'
    SOMETIMES = 'sometimes'

    @staticmethod
    def getall():
        return [Action.INSERT, Action.SUBSTITUTE, Action.SWAP, Action.DELETE, Action.SPLIT,
                Action.SEQUENTIAL, Action.SOMETIMES]