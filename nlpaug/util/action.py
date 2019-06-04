class Action:
    INSERT = 'insert'
    SUBSTITUTE = 'substitute'
    DELETE = 'delete'
    SWAP = 'swap'

    SEQUENTIAL = 'sequential'
    SOMETIMES = 'sometimes'

    @staticmethod
    def getall():
        return [Action.INSERT, Action.SUBSTITUTE, Action.SWAP, Action.DELETE, Action.SEQUENTIAL, Action.SOMETIMES]