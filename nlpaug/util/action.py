class Action:
    INSERT = 'insert'
    SUBSTITUTE = 'substitute'
    DELETE = 'delete'

    SEQUENTIAL = 'sequential'
    SOMETIMES = 'sometimes'

    @staticmethod
    def getall():
        return [Action.INSERT, Action.SUBSTITUTE, Action.DELETE, Action.SEQUENTIAL, Action.SOMETIMES]